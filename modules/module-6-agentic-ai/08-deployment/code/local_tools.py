"""In-process tools that stand in for the MCP server inside the container.

In class 5.5 (MCP) the glossary was an MCP server, started as a subprocess and reached
over stdio. That server imports the retriever, so inside a container it would load a
SECOND copy of the embedding model in a second Python process: on 512 MB that is the
difference between fitting and being killed.

LocalBridge has the same two methods the worker uses from McpBridge, specs() and
call(name, args), so the worker code does not change. This is the general lesson: MCP
is an interface, and a deployment can satisfy the same interface in-process when the
tools are yours and memory is tight. A third-party MCP server would stay a separate
service.
"""

from __future__ import annotations

GLOSSARY = {
    "hsa": "HSA (Health Savings Account): a tax-advantaged account for medical expenses, "
           "available with a high-deductible health plan.",
    "standard deduction": "Standard deduction: a fixed amount you subtract from income if "
                          "you do not itemize; it varies by filing status.",
    "agi": "AGI (Adjusted Gross Income): gross income minus specific adjustments; many "
           "limits and phase-outs are based on it.",
}


def tax_glossary(term: str) -> str:
    return GLOSSARY.get(term.strip().lower(), f"No glossary entry for {term!r}.")


class LocalBridge:
    """Duck-typed replacement for McpBridge: specs() and call()."""

    _SPECS = [{"type": "function", "function": {
        "name": "tax_glossary",
        "description": "Define a tax term in plain language (for example HSA, AGI, standard deduction).",
        "parameters": {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]}}}]

    def specs(self) -> list[dict]:
        return self._SPECS

    def call(self, name: str, args: dict) -> str:
        if name == "tax_glossary":
            return tax_glossary(**args)
        raise ValueError(f"unknown tool {name!r}")
