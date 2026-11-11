# Autonomous Research Agent

A small command-line research assistant built around native LLM tool calling. The
model chooses which tools to use, can make additional calls when the first results
are weak, and decides when it has enough information to answer. The application does
not prescribe a fixed search sequence.

## What it does

- Accepts a research question and asks for clarification when an ambiguity matters.
- Selects tools dynamically through function calling.
- Searches the public web with Tavily, searches Wikipedia and arXiv, and reads public
  webpages for more detail.
- Keeps source URLs in a registry and assigns stable citation IDs such as `[S1]`.
- Marks unknown citation IDs and warns when sources were retrieved but not cited inline.
- Preserves conversation messages and source IDs for follow-up questions during the
  current CLI session.
- Uses the machine's local clock for current-date, current-time, and relative-date
  questions when the model invokes the clock tool.
- Produces Markdown research reports and saves sourced responses in `reports/`.
- Lets the model decide when research is sufficient; a configurable maximum round count
  is only a safety ceiling.
- Compacts older tool results in large conversations and uses a shorter evidence packet
  for final synthesis.
- Recovers from late provider timeouts with a compact synthesis attempt or an explicit
  evidence-only report, preserving sources already collected.

## Architecture

```text
main.py
  └── ResearchAgent
        ├── LLMProvider (interface)
        │     └── OpenAICompatibleProvider (HTTP adapter)
        ├── ToolRegistry
        │     ├── web_search       (Tavily API)
        │     ├── wikipedia_search (MediaWiki API)
        │     ├── arxiv_search     (arXiv API)
        │     ├── fetch_webpage    (HTTP + BeautifulSoup)
        │     ├── calculate        (restricted arithmetic evaluator)
        │     └── get_current_datetime (machine-local clock)
        └── SourceStore           (stable citation IDs)
```

The agent depends on the `LLMProvider` interface, not on a vendor SDK. The included
adapter uses the OpenAI-compatible chat-completions format. To support a provider with
a different protocol, implement `LLMProvider.complete()` in a separate adapter; the
agent and tools do not need to change.

## Setup

Python 3.10 or newer is recommended.

```powershell
# 1. Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Create the local configuration file
Copy-Item .env.example .env
```

Edit `.env` with the exact OpenAI-compatible endpoint, API key, and model identifier
provided by your LLM service:

```dotenv
LLM_BASE_URL=https://your-provider.example/v1
LLM_API_KEY=replace_with_your_api_key
LLM_MODEL=replace_with_a_tool_calling_model
TAVILY_API_KEY=
REQUEST_TIMEOUT=45
LLM_TIMEOUT=240
MAX_TOOL_ROUNDS=12
```

`LLM_BASE_URL` is the base URL; the application appends `/chat/completions`. Do not
append `/chat/completions` yourself. Some providers require a `/v1` path and others
may document a different base path, so use the exact value from your provider's API
documentation. `LLM_MODEL` must be a model name or routing alias accepted by that
endpoint and the selected model must support tool/function calling.

### Using FreeLLMAPI

Keep the same provider-neutral code. Set `LLM_BASE_URL`, `LLM_API_KEY`, and
`LLM_MODEL` to the values supplied by the FreeLLMAPI endpoint or dashboard you use.
Do not copy an endpoint from an unrelated deployment; hosted and self-hosted setups
can use different URLs and model identifiers.

`TAVILY_API_KEY` is optional. The `web_search` tool requires it. Wikipedia, arXiv,
webpage reading, calculations, and the local clock do not require a Tavily key.

## Run

```powershell
python main.py
```

Enter a question at the `you>` prompt. Follow-up questions share the current
conversation context. Use `:reset` to start a new conversation, `:help` for commands,
and `:quit` to exit. Markdown reports backed by sources are saved in `reports/`.

Example prompts:

- `Compare retrieval-augmented generation and long-context approaches for enterprise question answering.`
- `Find recent arXiv work on small language models for edge devices. Summarize the main trade-offs.`
- `What are the main security risks in browser-based AI agents? Focus on primary sources.`
- `Can you expand the second finding and explain how it affects a small engineering team?`
- `What is today's date? Use the date tool.`

## Tests

The tests use a fake model provider and mocked HTTP responses, so they do not need
API keys or live network access.

```powershell
python -m pytest -q
```

## Troubleshooting tool calling

The CLI prints the names of tools called after each answer. If it prints that no tools
were called, the model returned a normal text response instead of a function call.
Check that:

1. The configured endpoint is the provider's chat-completions base URL.
2. `LLM_MODEL` names a model or router option that supports tool/function calling.
3. The API key and account have access to that model.
4. The provider accepts OpenAI-compatible `tools` and `tool_choice: "auto"` fields.

To test the clock path, ask `What is today's date? Use the date tool.` The date tool
reads the timezone configured on the computer running the agent. It does not infer
your location or timezone from your account.

## Notes and limitations

- The included LLM adapter targets OpenAI-compatible chat-completions APIs. A provider
  with a different wire format needs its own adapter implementing `LLMProvider`.
- Tavily search requires a Tavily API key. Wikipedia, arXiv, and webpage reading do not.
- Webpage extraction is text-focused; it is not a browser and does not execute
  JavaScript or log in to websites.
- Search results can be incomplete or stale. Verify important claims against source
  pages, especially when freshness or accuracy matters.
- Citation IDs are checked against the source registry, but the application cannot
  independently prove that each cited source supports the nearby claim.
- Conversation history is held in memory for the current CLI run; it is not a durable
  chat database.

## Terminal progress indicator

While a request is running, the CLI displays an animated terminal spinner with
short status updates such as `Thinking and deciding what to do`, `Running tool:
web_search`, and `Writing the final response and checking citations`. The indicator
uses a simple ASCII animation for compatibility with Windows PowerShell. When
output is non-interactive, it prints status changes as normal lines instead.


### Terminal progress display

The CLI shows a single-line animated progress indicator while the agent is working. Progress labels are clipped to the current terminal width to prevent carriage-return animation from wrapping and leaving duplicated fragments in Windows PowerShell. When output is redirected, progress updates are printed as normal log lines.


## Troubleshooting slow model responses

Long research prompts can require several model calls. The default `LLM_TIMEOUT=240` allows up to four minutes for each model response, while `REQUEST_TIMEOUT=45` applies to external research tools. See the sections below for the distinction between the adaptive research loop, its safety ceiling, and timeout recovery. The terminal spinner displays the current activity while the agent works.

## How the research loop and round budget work

The model decides when it has enough information. It can answer immediately with zero
research rounds, use a few rounds, or continue searching across more rounds. The
`MAX_TOOL_ROUNDS` value is only a safety ceiling; it is not a target and the agent does
not deliberately use every round. The default is 12, and values from 1 to 30 are
accepted. Increase it for broad investigations or lower it for quick lookups.

A round means one model decision that returns one or more tool calls. It does not mean
one individual search result. When the safety ceiling is reached, the agent asks for a
final answer using a compact evidence packet rather than sending the entire tool-call
transcript back to the model.

To change the limits, edit the existing `.env` file (not just `.env.example`):

```dotenv
LLM_TIMEOUT=240
MAX_TOOL_ROUNDS=12
```

`LLM_TIMEOUT` is the maximum wait for an individual model response. It is separate from
`REQUEST_TIMEOUT`, which applies to external research websites. Existing `.env` files
keep their old values when you update the code, so edit them manually if they still say
`LLM_TIMEOUT=180` or `MAX_TOOL_ROUNDS=7`.

## Large research requests and timeout recovery

Tool results are kept as valid JSON but shortened before they are sent back to the model.
If the active conversation becomes large, older tool replies are compacted and a short
source index is supplied so the model can still cite earlier findings. The full transcript
and source registry remain available to the running application.

If a provider call times out after the agent has collected sources, the agent makes one
smaller, tool-free synthesis request. If that also times out, it returns an evidence-only
report with the collected titles, URLs, and snippets instead of discarding the research.
That fallback is clearly labelled as incomplete; it is not presented as a finished
comparative analysis. If the very first model request times out before any evidence is
collected, the timeout is shown so the provider settings can be checked.

A longer timeout cannot override a provider's own server-side generation limit, rate
limit, account quota, or model availability. If timeouts continue, check the provider
status, model identifier, endpoint, rate-limit responses, and the model's tool-calling
support. A faster tool-calling model or provider may help more than continually raising
the timeout.
