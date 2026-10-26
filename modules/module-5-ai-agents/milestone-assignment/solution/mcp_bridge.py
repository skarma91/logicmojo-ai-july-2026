"""A synchronous bridge to an MCP server (class 5.7).

The MCP client from class 5.5 is async (stdio + asyncio). Our assembled agent runs on
a synchronous LangGraph loop. This bridge reconciles the two: it owns one asyncio event
loop on a background thread, keeps a single MCP session open for the agent's lifetime,
and exposes two plain synchronous methods:

  - specs():        the discovered tools as OpenAI-style specs (for chat_tools),
  - call(name, a):  invoke a discovered tool over the protocol and return its text.

This is the real pattern for embedding an async MCP client inside a sync application:
marshal each call onto the persistent loop with run_coroutine_threadsafe. Tool RESULTS
are untrusted (class 5.6); the caller scans them.

Install:  pip install "mcp<2"
"""

from __future__ import annotations
import asyncio
import logging
import pathlib
import sys
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger("course.mcpbridge")
SERVER = str(pathlib.Path(__file__).parent / "server.py")


def _result_text(result) -> str:
    return "\n".join(getattr(c, "text", "") or "" for c in result.content).strip()


class McpBridge:
    """Launches server.py once, discovers its tools, and calls them synchronously."""

    def __init__(self, server: str = SERVER):
        self._server = server
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._session: ClientSession | None = None
        self._tools = []
        self._ready = threading.Event()
        self._serve_future = None

    # -- lifecycle ---------------------------------------------------------
    # The MCP session (anyio task group) must be opened AND closed in the same task.
    # So one long-lived coroutine (_serve) owns the whole lifecycle: it connects,
    # then waits for a shutdown signal, then tears down. Calls are marshaled onto the
    # same background loop, and teardown just sets the shutdown event.
    def __enter__(self) -> "McpBridge":
        self._thread.start()
        self._serve_future = asyncio.run_coroutine_threadsafe(self._serve(), self._loop)
        if not self._ready.wait(timeout=30):
            raise RuntimeError("MCP server did not become ready")
        return self

    def __exit__(self, *exc):
        self._loop.call_soon_threadsafe(self._stop.set)
        try:
            self._serve_future.result(timeout=10)
        except Exception as e:
            log.warning("bridge shutdown: %s", e)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def _run(self, coro):
        """Submit a coroutine to the background loop and wait for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _serve(self):
        self._stop = asyncio.Event()          # bind to the background loop's context
        params = StdioServerParameters(command=sys.executable, args=[self._server])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()                       # the MCP handshake
                self._session = session
                self._tools = (await session.list_tools()).tools  # discovery
                log.info("discovered MCP tools: %s", [t.name for t in self._tools])
                self._ready.set()
                await self._stop.wait()                          # stay open for calls
        self._session = None

    # -- the sync surface the agent uses -----------------------------------
    def specs(self) -> list[dict]:
        return [{"type": "function", "function": {
            "name": t.name,
            "description": t.description or "",
            "parameters": t.inputSchema or {"type": "object", "properties": {}}}}
            for t in self._tools]

    def names(self) -> set:
        return {t.name for t in self._tools}

    def call(self, name: str, args: dict) -> str:
        return self._run(self._call(name, args))

    async def _call(self, name: str, args: dict) -> str:
        return _result_text(await self._session.call_tool(name, args))
