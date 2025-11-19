# RAG

## Retrieval Algorithms

1. term-based retrieval

    Using keywords. 

    Elasticsearch / BM25

2. embedding-based retrieval 

    Embed both query and external knowledge. It has:

    a. Embedding model

    b. Retriever: fetch relevant embeddings

    For larger scale search: ANN: FAISS, ScaNN, Hnswlib
    
    Rank of retrieved documents: NDCG (normalized discounted cumulative gain), MAP (Mean Average Precision), and MRR (Mean Reciprocal Rank)

    Evaluation:

        1. Retrieval quality

        2. Final RAG outputs

        3. Embeddings

3. hybrid retrieval

## Retrieval Optimization

1. Chunking

    Simple way: equal length

    Set overlapps

2. Reranking

    Can be based on time

3. Query Rewriting

4. Contextual retreval

    Augment chunk with relevant context - metadata

    Especially in chunks, key info could be missing. Can use the original document's title / summary

## Multi-Modality

Image: CLIP

Process both image and text with CLIP and compare. For queries, do CLIP on them too. 

Tables: Text-2-SQL

# 🧠 Agentic RAG Overview

## 1. What is RAG (Retrieval-Augmented Generation)

**Goal:** Improve LLM accuracy and factual grounding by retrieving external knowledge before generation.

**Basic pipeline:**
1. **User Query → Embedding**
2. **Vector Search → Retrieve Relevant Documents**
3. **LLM Input = [Query + Retrieved Context]**
4. **LLM → Generate Answer**

**Limitations of standard RAG:**
- Cannot reason across multiple steps or sources.
- Only retrieves text; cannot perform actions.
- Cannot plan or retry if retrieval quality is poor.
- Not adaptive to complex user intents.

---

## 2. What is an Agentic RAG

**Agentic RAG = RAG + Planning + Tool-Use**

It extends the standard RAG pipeline by allowing the LLM to:
- **Plan** what steps to take (reason about the task)
- **Act** by calling retrieval tools, APIs, or custom functions
- **Iterate** on results until the task is complete

In other words, the LLM becomes an **Agent** that can **decide**, **retrieve**, and **execute** — not just generate.

---

## 3. Core Components of Agentic RAG

| Component | Description | Example |
|------------|-------------|----------|
| **Planner** | Decides what actions or tools to use, and in what order. | “Step 1: Retrieve company policy → Step 2: Summarize → Step 3: Send email.” |
| **Tools** | Executable functions or APIs that the LLM can call. | `vector_search()`, `sql_query()`, `send_email()` |
| **Memory (optional)** | Keeps track of previous steps, retrieved results, and user context. | “I already summarized this file earlier.” |
| **Executor** | Runs each action proposed by the Planner. | Executes `RAG()` or `API()` and returns the result. |

---

## 4. Why Agentic RAG Matters for Business Projects

In a real-world RAG system (like your company project), users often ask:
> “Summarize the latest sales report and include the CEO’s comments.”

This requires:
1. Retrieving structured data (from a SQL database)
2. Retrieving text (from reports via vector search)
3. Combining and reasoning over both
4. Possibly sending results or triggering an action

A **pure RAG** cannot coordinate these steps.  
An **Agentic RAG** can plan and execute them automatically.

---

## 5. Example Workflow
User Query → Planner → Retrieve → Generate → Act → Final Output


**Step-by-step example:**
1. Planner detects user intent (“summarize + analyze”)
2. Retrieves documents using vector database
3. Optionally queries external APIs (e.g., financial or CRM data)
4. Synthesizes results and generates structured output
5. Executes optional actions (e.g., save report, send email)

---

## 6. Implementation Strategy for Your RAG Project

### (1) Start with Core RAG
- Use **LangChain** or **LlamaIndex** to implement vector retrieval.  
- Example backends: FAISS, Chroma, or Azure Cognitive Search.

### (2) Add Tool-Use Capabilities
- Define tools (functions) for:
  - Document retrieval
  - SQL query execution
  - API calls
- Register them via **OpenAI Function Calling** or **LangChain Tools**.

### (3) Add a Planner Layer
- Implement reasoning using **ReAct** or **Plan-and-Execute** patterns.  
- Let the LLM plan a sequence of steps dynamically.

### (4) Integrate into FastAPI
- Build endpoints like `/query`, `/retrieve`, `/act`
- Let the Agent orchestrate the workflow per user request.

---

## 7. Evaluation & Optimization

When evaluating your Agentic RAG pipeline, consider:
- **Retrieval quality:** relevance and coverage of returned documents
- **Planning reliability:** does the Agent choose correct tools?
- **Latency:** planning adds overhead, optimize caching and batching
- **Scalability:** vector DB performance, parallel retrieval
- **Security:** prevent the Agent from executing unsafe actions

---

## 8. Example Frameworks & Tools

| Category | Tools/Frameworks |
|-----------|------------------|
| Vector Search | FAISS, Chroma, Pinecone, Azure Cognitive Search |
| Agent Frameworks | LangChain, LlamaIndex, DSPy |
| LLM Providers | OpenAI (GPT-4), Anthropic (Claude), Mistral |
| API Integration | FastAPI, Function Calling, Azure OpenAI Service |

---

## 9. Key Takeaways

- **RAG** grounds LLMs with factual retrieval.  
- **Agentic RAG** empowers them with reasoning, planning, and tool-use.  
- For enterprise projects, Agentic RAG bridges the gap between **data retrieval** and **business automation**.  
- The **Planner** acts as the decision-making layer, enabling dynamic workflows.  

---

## 10. Next Steps for Your Project

1. ✅ Review your current RAG pipeline (retrieval + generation).  
2. 🧩 Define potential tools (e.g., document search, database access, API calls).  
3. 🧠 Prototype a Planner Agent using LangChain’s `PlanAndExecute` or OpenAI’s `tool_choice="auto"`.  
4. ⚙️ Test with real user queries from your business context.  
5. 📈 Measure accuracy, reasoning trace quality, and performance.

---

**Goal:**  
Transform your existing RAG system into an **Agentic Knowledge Assistant** capable of autonomous retrieval, reasoning, and action.

