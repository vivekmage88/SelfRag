# Self-correcting RAG — LangGraph

A retrieval-augmented generation pipeline built as a state graph rather than a linear chain. Retrieved documents are graded for relevance by an LLM; if none survive, the question is rewritten and retrieval runs again, up to a retry limit.

Retrieves from the PostgreSQL + pgvector store built in [`pgrag`](../pgrag).

---

## The graph

```
START → retrieve → grade → ┬→ generate → END
                           ├→ rewrite → retrieve   (cycle)
                           └→ give_up → END
```

Three exits, one loop back to a node that has already run. The backwards edge is the thing a chain cannot express, and the reason this is a graph.

| File | Holds | Why separate |
|---|---|---|
| `state.py` | The `State` TypedDict | Imported by everything, depends on nothing |
| `nodes.py` | Five node functions and one decision function | The work. Each testable alone with a plain dict. |
| `graph.py` | The wiring only | Structure separate from behaviour — re-wire without touching logic |

---

## The three concepts

**State** — one dict that exists for the whole run. Nodes do not pass data to each other; they all read and write the same shared object.

```python
class State(TypedDict):
    question: str            
    original_question: str   
    chunks: list             
    generation: str         
    retry_count: int         
```

**Node** — a plain function taking the state and returning **only the fields it changed**. LangGraph merges that dict into the running state.

```python
def retrieve_node(state: State) -> dict:
    ...
    return {"chunks": matches}
```

**Decision function** — takes the state, changes nothing, returns **a string** naming the next node.

```python
def decide_after_grading(state: State) -> str:
    if state["chunks"]:
        return "generate"
    if state["retry_count"] >= MAX_RETRIES:
        return "give_up"
    return "rewrite"
```

That is the whole distinction: a node *does* something, a decision function *chooses* something.

Wired up, the decision becomes a forking edge:

```python
builder.add_conditional_edges(
    "grade",                                        
    decide_after_grading,                            
    {"generate": "generate",                          
     "rewrite": "rewrite",                            
     "give_up": "give_up"},
)
builder.add_edge("rewrite", "retrieve")         
```

---

## Why grade at all when retrieval already filters by distance

Distance measures **similarity**, not **usefulness**. A chunk about application lifespan sits close to a question about background tasks — adjacent topic, overlapping vocabulary — without answering it. A number cannot tell the difference; an LLM reading the text can.

Grading uses structured output rather than parsed text:

```python
class Grade(BaseModel):
    relevant: bool
    reason: str

grader = model.with_structured_output(Grade)
```

`result.relevant` is a real boolean. The alternative — prompting for "yes or no" and parsing `"Yes, this document is relevant because…"` — breaks the first time the model rephrases. Under the hood this is the same tool-calling mechanism the model uses for functions: the Pydantic model becomes a JSON schema passed as a tool definition.

`reason` is never used in code. It is printed, so a run shows *why* each chunk was kept or dropped.

---

## Run traces

**Answerable question** — straight through, five node executions:

```
[retrieve] 1 chunks for: how do background tasks work?
[grade] p17: True — The document explains how background tasks work...
[grade] kept 1 of 1
[decide] have chunks -> generate
[generate] 612 chars
```

**Unanswerable question** — the cycle runs twice, then the bound stops it. Eleven node executions, `retrieve` runs three times:

```
[retrieve] 0 chunks for: what is the capital of France?
[decide] nothing relevant -> rewrite
[rewrite] 'what is the capital of France?' -> 'What is the administrative center of France?'
[retrieve] 0 chunks
[decide] nothing relevant -> rewrite
[rewrite] -> 'What is the capital city of France?'
[retrieve] 0 chunks
[decide] out of retries -> give up
```


## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # DATABASE_URL and OPENAI_API_KEY

python nodes.py          # test nodes individually
python graph.py          # run the full graph
```

Requires an indexed document in the `pgrag` database.
