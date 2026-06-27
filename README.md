# Knowledge Base Q&A System

A **Corrective RAG (CRAG)** pipeline that answers questions over a document knowledge base. Built with LangChain and LangGraph, it goes beyond naïve retrieval by grading every retrieved chunk for relevance and automatically rewriting the query when the retrieval misses.

## How It Works

```
START → retrieve → grade_documents ─┬─► generate → END
             ▲                      │
             │                      └─► transform_query ─┘
             └─────────────────────────────────────────────
```

| Node | Role |
|---|---|
| `retrieve` | Fetches the top-k chunks from the vector store |
| `grade_documents` | LLM scores each chunk for relevance; irrelevant chunks are dropped |
| `generate` | Produces the final answer from the remaining relevant chunks |
| `transform_query` | Rewrites the question to improve retrieval when no relevant chunks are found |

If `grade_documents` finds zero relevant chunks, the pipeline loops back through `transform_query → retrieve` (up to 2 retries) before generating a graceful "not enough information" response.

## Tech Stack

| Component | Choice |
|---|---|
| Language | Python 3.13 |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | LangChain `InMemoryVectorStore` |
| Text splitting | `RecursiveCharacterTextSplitter` |
| Persistence | LangGraph `MemorySaver` checkpointer |

## Project Structure

```
.
├── notebook/
│   └── knowledge_base_qa.ipynb   # Full CRAG prototype with examples and output
├── main.py                        # Entry point (WIP)
├── pyproject.toml
└── .env                           # OPENAI_API_KEY (not committed)
```

## Setup

**Prerequisites:** Python 3.13, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/samruk-code/knowledge_base_Q-A_answering_system.git
cd knowledge_base_Q-A_answering_system

# Install dependencies
uv sync

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env
```

## Running the Notebook

```bash
uv run jupyter notebook notebook/knowledge_base_qa.ipynb
```

The notebook walks through:
1. Building the vector store from sample documents
2. Constructing the LangGraph CRAG pipeline
3. Batch testing with in-scope and out-of-scope questions
4. Streaming mode to observe each graph node as it fires

## Example Output

```
QUESTION: What are the pricing plans and how much does Professional cost?

  [RETRIEVE] 4 docs fetched.
  [GRADE] 1/4 relevant.
  [GENERATE] Using 1 docs.

ANSWER:
TechBase Pro offers three plans:
  - Starter (Free) — up to 5 users, 10 GB storage, 1,000 queries/month
  - Professional — $49/user/month, unlimited users, 100 GB storage, SSO, priority support
  - Enterprise — custom pricing, dedicated infrastructure, on-premise option, 24/7 SLA support
```

Out-of-scope queries trigger the corrective loop and fall back gracefully:

```
QUESTION: What is the weather like today?

  [RETRIEVE] 0/4 relevant → [REWRITE] attempt 1
  [RETRIEVE] 0/4 relevant → [REWRITE] attempt 2
  [RETRIEVE] 0/4 relevant → max retries reached
  [GENERATE] Using 0 docs.

ANSWER:
The context does not provide information about current weather conditions.
```

## Key Design Decisions

- **Structured output grading** — the relevance grader uses `with_structured_output(GradeDocument)` (a Pydantic model) instead of parsing free-text, which eliminates prompt-injection risk and makes scoring deterministic.
- **Retry cap** — the rewrite loop is bounded at 2 attempts to prevent infinite loops on unanswerable questions.
- **MemorySaver checkpointing** — each query run is tied to a `thread_id`, enabling multi-turn conversation history within a session.
