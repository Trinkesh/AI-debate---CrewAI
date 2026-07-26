# AI Debate Club — Learn CrewAI + Groq

A tiny but genuinely illustrative multi-agent project: three AI agents
debate a topic and one of them judges the winner. It's designed to teach
you the core CrewAI concepts in ~100 lines of code.

## 1. Setup

```bash
pip install -r requirements.txt
```

Get a **free** Groq API key at https://console.groq.com/keys, then either:

```bash
export GROQ_API_KEY="gsk_..."       # recommended
```

or edit the placeholder directly inside `debate_crew.py`.

## 2. Run it

```bash
python debate_crew.py
```

You'll see live streaming output as each agent "thinks" and writes,
followed by a final verdict block.

## 3. The core concepts (in the order you'll meet them in the code)

### Agent
An `Agent` is a persona, not a fixed prompt. You give it a `role`, a
`goal`, and a `backstory`; CrewAI assembles these into a system prompt
for you. This is the opposite of writing one giant prompt by hand — you
compose behavior out of small, reusable persona definitions.

### LLM
CrewAI doesn't talk to Groq directly — it goes through **LiteLLM**, a
routing layer that understands dozens of providers. That's why the model
string is `"groq/llama-3.3-70b-versatile"` instead of just
`"llama-3.3-70b-versatile"` — the `groq/` prefix tells LiteLLM which
provider's API to call and which API key env var to read
(`GROQ_API_KEY`).

### Task
A `Task` pairs work (`description`) with an expectation
(`expected_output`) and assigns it to one `Agent`. Splitting work into
tasks — rather than one big "do everything" agent — is what lets you
mix specialists.

### context (the important part)
```python
judge_task = Task(..., context=[pro_task, con_task])
```
This is CrewAI's mechanism for passing information *between* agents. The
judge doesn't magically know what the debaters said — `context` tells
CrewAI to inject `pro_task` and `con_task`'s completed outputs into the
judge's prompt before it runs. This is the pattern you'll reuse in every
real CrewAI project: researcher → writer → editor, scraper → analyst →
report-writer, etc.

### Crew + Process
```python
crew = Crew(agents=[...], tasks=[...], process=Process.sequential)
```
The `Crew` is the orchestrator. `Process.sequential` runs tasks in the
listed order, each with access to prior tasks' outputs (if wired via
`context`). The other built-in option, `Process.hierarchical`, adds a
manager agent that dynamically decides which agent handles what — worth
exploring once this version feels familiar.

## 4. Ideas to extend it (good next exercises)

- **Add a `Tool`**: give the debaters a web-search tool (via
  `crewai-tools`) so their arguments cite real, current facts.
- **Make it interactive**: read `TOPIC` from `input()` or `sys.argv`
  instead of hardcoding it.
- **Add a rebuttal round**: add `pro_rebuttal` / `con_rebuttal` tasks
  with `context=[con_task]` / `context=[pro_task]` before the judge
  weighs in.
- **Swap the model**: try `groq/llama-3.1-8b-instant` for a much faster
  (but less nuanced) debate, and compare quality.
- **Switch to `Process.hierarchical`**: add a "Debate Moderator" manager
  agent that decides dynamically how many rounds to run.

## 5. Why Groq specifically?

Groq runs open models (Llama, etc.) on custom LPU hardware, so inference
is extremely fast — noticeable when you're watching multiple agents
generate output in sequence, as here. The free tier's rate limits are
generous enough for experimenting with multi-agent projects like this
one without hitting a paywall.
