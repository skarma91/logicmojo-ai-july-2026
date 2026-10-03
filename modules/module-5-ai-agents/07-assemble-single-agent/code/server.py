"""Class 5.5 build: a custom MCP server.

MCP (the Model Context Protocol) is a common protocol so that any MCP-speaking
CLIENT can use any MCP SERVER's tools, no bespoke integration per app. Here we build
the SERVER: it wraps our own functions as tools and exposes them over stdio. The
agent (client.py) connects to it and discovers these tools through the protocol.

This uses the standard `mcp` SDK's FastMCP. Each @mcp.tool() function becomes an MCP
tool; its name, docstring (description), and typed signature become the tool's schema
automatically, which is exactly what a client needs to decide when to call it.

Scope: we build TOOLS over stdio, at build-and-ship depth. MCP also has resources
(read-only data) and prompts (reusable templates), and other transports (HTTP/SSE);
those are named in the slides as further scope, not built here.

Install and run (the client launches this automatically; you rarely run it by hand):
    pip install "mcp<2" sentence-transformers numpy
    python server.py
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

import retriever   # the Module 4 retriever

mcp = FastMCP("tax-tools")


@mcp.tool()
def search_tax_docs(query: str) -> str:
    """Search official tax publications for a fact (limits, deductions, amounts)."""
    hits = retriever.search(query, k=3)
    return "\n".join(f"[{h['id']}] {h['text']} (source: {h['pub']} {h['tax_year']}, p.{h['page']})"
                     for h in hits)


# A pure-Python tool (no model needed), so the protocol handshake is easy to try.
_GLOSSARY = {
    "hsa": "HSA (Health Savings Account): a tax-advantaged account for medical expenses, "
           "available with a high-deductible health plan.",
    "standard deduction": "Standard deduction: a fixed amount you subtract from income if "
                          "you do not itemize; it varies by filing status.",
    "agi": "AGI (Adjusted Gross Income): gross income minus specific adjustments; many "
           "limits and phase-outs are based on it.",
}


@mcp.tool()
def tax_glossary(term: str) -> str:
    """Define a tax term in plain language (for example HSA, AGI, standard deduction)."""
    return _GLOSSARY.get(term.strip().lower(), f"No glossary entry for {term!r}.")


if __name__ == "__main__":
    mcp.run()   # stdio transport by default
