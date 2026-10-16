# Class 6.3 code: context engineering, measured

`main.py` runs the class 6.2 (Orchestration) team (two parallel researchers, a writer, a reviewer
loop) on the same questions twice, then compares what each run sent to the model.

| | baseline | engineered |
|---|---|---|
| What the writer and reviewer see | the full shared state: every researcher's raw transcript (tool calls and untrusted tool output), every earlier draft and verdict | only what their job needs: the question, the handoff notes, and (reviewer) the draft |
| Researcher handoff | raw transcript | a structured note: cited facts, verified source ids, open questions |
| Prompt order | a per-request id and time on the first line, then everything else | stable prefix first (shared team rules, then the role), variable content last |
| Retrieval | every lookup hits the retriever | memoized across the whole team |

The team state holds everything in both runs. Only the **per-agent context
functions** (`writer_messages`, `reviewer_messages`) differ. Holding data and
showing it to a model are separate decisions.

## What the report shows

- **Prompt tokens per agent**, with the cached part in brackets. These are the
  provider's own `usage` numbers (`prompt_tokens`, and `cached_tokens` from
  `usage.prompt_tokens_details`), recorded by the Module 6 `llm.py`. Nothing is
  estimated.
- **Cost** at illustrative prices, with cached input billed at half price, and the
  cache saving $\Delta C = c \cdot p_{in} \cdot 0.5$.
- **Handoff compression**: raw transcript tokens, note tokens, and the ratio
  $r = T_{handoff} / T_{raw}$ for every researcher. The note also lists any id the
  researcher cited but never retrieved (`unverified`).
- **Writer budget** for the first round, split as
  $T = T_{system} + T_{tools} + T_{context} + T_{task}$. The context term is where
  the two runs differ.
- **Quality**: the final answer must cite the expected source and pass the exact
  checks (word limit, and every cited id was actually retrieved). Saving tokens
  only counts if this still passes.
- **Retrieval memo**: how many lookups were served from the memo. It only helps when
  agents repeat a query, so a low count is normal on a two-question run.

Handoff and budget sizes are counted locally with `tiktoken` (`o200k_base`, the
gpt-oss family's base vocabulary). If `tiktoken` is not installed, the script falls
back to characters divided by four and says so in the report.

## What to expect from caching

Groq caches prompt prefixes automatically on gpt-oss models: a request that starts
with exactly the same tokens as a recent one has that prefix served from cache, at
half price and without counting toward rate limits (cached tokens are still subtracted
after processing, so large parallel bursts can hit the cap anyway). Hits are likely
but not guaranteed, and a cached prefix expires after 2 hours without use. Three
things follow, and you can see each one in the report:

- **The baseline caches almost nothing.** Its first line changes on every request,
  and everything after a changed token is a cache miss.
- **The engineered writer and reviewer cache their stable prefix** once they repeat
  it (a second round, or a second question).
- **Short prefixes do not cache.** The minimum cacheable length depends on the model
  (roughly 128 to 1024 tokens). If an agent's stable prefix is below it,
  `cached_tokens` stays 0 for that agent. The fix is not to pad the prompt; the
  saving simply does not apply there.

The researcher worker is the same in both runs. Its own context grows inside its
tool loop (class 5.3, Memory for agents), and that is a separate lever.

## Free-tier note

Both questions in both modes make about 30 calls, and the baseline's prompts are
large. On Groq's free plan (at time of writing about 30 requests and 8K tokens per
minute for `gpt-oss-20b`) expect a few automatic 429 retries; `llm.py` waits and
retries. Use `--questions 1` to halve the run. Ollama has no rate limit, but local
servers usually report no cached tokens.

## What ships here

```
main.py          baseline versus engineered, the context functions, the report
tax_worker.py    the class 5.7 (Assemble a single agent) agent as a subgraph; 6.3 adds search_fn and return_transcript
llm.py           Module 6 wrapper: retry on 429, usage (including cached tokens), per-agent attribution
retriever.py, guardrails.py, mcp_bridge.py, server.py, data/corpus.jsonl
```

## Run

```
pip install langgraph "mcp<2" sentence-transformers numpy python-dotenv groq tiktoken
# set PROVIDER=groq and GROQ_API_KEY in .env (or PROVIDER=ollama)
python main.py                  # both questions
python main.py --questions 1    # one question
```
