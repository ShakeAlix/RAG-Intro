# RAG Intro

A retrieval-augmented generation pipeline built from scratch over a single PDF, with a hybrid sparse/dense retriever and a real evaluation harness comparing them.

## Overview

This is my first hands-on RAG project. I wanted to actually understand retrieval-augmented generation — chunking tradeoffs, what an embedding is doing, what BM25 is scoring, how rank fusion works — rather than copy a tutorial's chain and call it done. Every phase below started as a concept I had to look up and implement myself, not boilerplate from a template.

To force real decisions instead of a trivial "load PDF, embed, done" pipeline, I picked a source document on purpose: Wikipedia's [List of HTTP status codes](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes) article, saved as a PDF (`files/http_codes.pdf`). It mixes two very different content types in the same document — short exact tokens like `ERR-4471`-style codes (`403`, `417`, `429`) and long descriptive prose explaining *why* a server would return them. That split is exactly what motivates combining sparse and dense retrieval instead of picking one, and it's what the evaluation section is built to measure.

## Architecture

The pipeline runs as a sequence of stages, each a section in [main.ipynb](main.ipynb):

1. **Ingestion** — `PyPDFLoader` loads the PDF into LangChain `Document` objects, one per page, each carrying `page_content` and page-number metadata.
2. **Chunking** — `RecursiveCharacterTextSplitter` splits each page into overlapping chunks, trying paragraph → line → word boundaries before falling back to hard splits. Each chunk gets a stable `chunk_id` (page number + an MD5 hash of its content) so it can be persisted and re-referenced without duplication across runs.
3. **Dense indexing** — every chunk is embedded with a Gemini embedding model and stored in a persistent Chroma vector database, queried by cosine similarity through a standard LangChain retriever.
4. **Sparse indexing** — the same chunks are indexed into a BM25 retriever, but with a custom preprocessing step (below) instead of BM25's default tokenizer.
5. **Hybrid fusion** — the dense and sparse retrievers are combined with LangChain's `EnsembleRetriever`, which merges their rankings via Reciprocal Rank Fusion (RRF) rather than a naive score average.
6. **Generation** — the fused retriever's results are formatted into a context block and dropped into a prompt template alongside the user's question, following the LCEL "stuff" pattern, and sent to a Gemini chat model.
7. **Evaluation** — a hand-labeled set of test queries with known correct pages is run through BM25-only, dense-only, and hybrid retrieval, scored with standard IR metrics, and visualized.

## Tech Stack

Pulled directly from the notebook's imports and installs, not from a generic RAG list:

- **Orchestration**: LangChain (`langchain`, `langchain_community`, `langchain_classic` for `EnsembleRetriever`)
- **Document loading**: `pypdf` via `PyPDFLoader`
- **Chunking**: `langchain_text_splitters.RecursiveCharacterTextSplitter`
- **Embeddings**: Google's `gemini-embedding-001` via `langchain_google_genai.GoogleGenerativeAIEmbeddings`
- **Vector store**: Chroma (`langchain_chroma`), persisted locally to `./my_chroma_db`
- **Sparse retrieval**: `langchain_community.retrievers.BM25Retriever` (backed by `rank_bm25`)
- **Sparse tokenizer**: spaCy (`en_core_web_sm`) — see below
- **Generation LLM**: Gemini chat model (`gemini-3.6-flash`) via `langchain_google_genai.ChatGoogleGenerativeAI`
- **Evaluation/analysis**: `pandas`, `matplotlib`, `numpy`

Note: `langchain-openai` was installed early on while I was still deciding on a provider, but the pipeline that actually runs end-to-end uses Google's Gemini models for both embeddings and generation — the OpenAI package is an unused leftover from that decision.

## Notable Implementation Details

- **Chunking**: `chunk_size=1000`, `chunk_overlap=100`, splitting on `["\n\n", "\n", " ", ""]` in priority order. This was picked empirically by eyeballing which size kept error codes attached to their explanatory sentence rather than severed across a chunk boundary. It's a single, uniform configuration across the whole document — I did not implement separate chunking rules for the code-dense sections vs. the prose sections (see *Learnings* below for why I'd revisit this).
- **Custom BM25 tokenizer**: instead of BM25's default whitespace/regex tokenizer, chunks are preprocessed with spaCy — lemmatized, lowercased, with stopwords and punctuation stripped — before being indexed and before each query is scored. This keeps BM25 from being thrown off by inflection and filler words in the wordier passages.
- **Stable chunk IDs**: each chunk's ID is derived from its page number plus an MD5 hash of its own text, so the same chunk gets the same ID even if the document is re-chunked, which lets the Chroma store be updated without duplicate entries.
- **Reload-without-re-embedding path**: the notebook includes a section that rebuilds both retrievers straight from the persisted `./my_chroma_db` store (via `db.get()`, which reads from disk, not from the embeddings API) instead of re-running ingestion. This means the evaluation section can be re-run repeatedly without spending API calls to re-embed the corpus — the only remaining Gemini calls are one embedding call per query, which dense/hybrid retrieval need to run at all.
- **Grounded prompt template**: the generation prompt explicitly instructs the model to answer *only* from the provided context, say it doesn't know rather than guess if the answer isn't there, and cite the page number(s) the answer came from — pushing citation and refusal behavior into the prompt rather than trusting the model's default behavior.

## Evaluation Methodology and Results

**Test set**: 11 hand-written queries split into four buckets — pure code lookups (`"417"`, `"403"`), pure conceptual/prose questions with no codes (`"how does content negotiation work"`), mixed queries referencing codes in a conceptual question (`"why would a WebDAV PROPFIND request fail with a 403 or 424"`), and a fourth bucket of ambiguous/reused codes (`"451"`, `"420"`, `"530"` — status codes with non-obvious or joke/informal histories). Ground truth (expected page number(s) per query) was determined by manually grepping the persisted chunk text, not by asking the LLM, so the metrics aren't circular.

**Metrics**: for each query, all three retrievers (BM25-only, dense-only, hybrid at equal 0.5/0.5 weighting) were run once at k=5, and Hit@5, Precision@5, Recall@5, and Mean Reciprocal Rank were computed from the same result set.

**Aggregate results:**

| Retriever | Hit Rate | Precision@5 | Recall@5 | MRR |
|---|---|---|---|---|
| BM25 | 1.000 | **0.436** | **1.000** | **0.882** |
| Dense | 1.000 | 0.400 | 0.939 | 0.833 |
| Hybrid | 1.000 | 0.400 | 0.970 | 0.818 |

**Honest read of this**: hybrid did *not* clearly beat BM25 alone at equal weighting on this corpus — BM25 leads on precision, recall, and MRR, and hybrid actually posts the lowest MRR of the three. All three retrievers hit 100% Hit@5, so the corpus is small enough that everything finds *something* relevant; the metrics that differentiate them are precision and rank quality, and there BM25 wins.

The likely reasons, rather than "hybrid doesn't work":
- **The corpus is small and topically narrow.** One 10-page document about one subject gives the dense embedding space less room to show its advantage — everything is already semantically close to everything else, so BM25's exact-match precision has less to lose against.
- **Several test queries are literal numeric codes** (`"417"`, `"403"`, `"451"`, `"420"`, `"530"`), which is exactly the case BM25 was designed to nail and dense embeddings tend to underweight (a bare number carries little semantic signal for an embedding model to latch onto). With roughly half the test set built from bare codes, the aggregate numbers are structurally tilted toward BM25's strength.
- Per-query breakdowns in the notebook (the `mixed` bucket comparison) do show dense contributing hits BM25 misses on purely conceptual phrasing — the benefit of hybrid is visible at the individual-query level even where it doesn't move the aggregate average.

This is run once at one weighting (0.5/0.5); I have not yet swept weight ratios or re-run with a larger/more varied query set, both of which I'd want before drawing a stronger conclusion either way.

## Learnings / What I'd Approach Differently

- **Uniform chunking left a decision on the table.** The whole premise of this project was that code-dense and prose-dense sections might need different chunking treatment, but I ended up shipping one `chunk_size`/`chunk_overlap` pair for the entire document. Splitting per-section (or chunking adaptively based on content density) is the most obvious thing I'd try first on a second pass.
- **Small ground-truth sets make aggregate metrics noisy.** With 11 queries, a single query flipping changes the aggregate percentages by ~9 points. I'd want a larger, more varied hand-labeled set before trusting small deltas between retrievers.
- **I only tested one hybrid weighting.** The whole point of Phase 7 was to let evaluation drive the sparse/dense weight choice empirically, and I didn't actually sweep it — I'd want to try a few weight ratios (e.g. 0.7/0.3, 0.3/0.7) against the same eval set before picking one.
- **BM25 winning wasn't the outcome I expected going in**, and figuring out *why* — rather than just reporting the number — was the most useful part of the exercise. It's a good reminder that "hybrid" isn't automatically better; it's only as good as the weighting and the corpus it's tuned against.

## Future Work

- **Semantic caching** — cache (query embedding → answer) pairs and check similarity against cached queries before hitting the LLM again, instead of calling it fresh on every query.
- **Cross-encoder reranking** on top of the hybrid results, since RRF fusion doesn't itself re-score relevance the way a reranker would.
- **Sweeping the hybrid fusion weights and chunk-size configurations** against the eval harness now that it exists, instead of picking both by eyeballing.
- Broader retrieval-augmented generation concepts beyond what this project covers — this was a first pass at the fundamentals, not a survey of the field.

## Setup and How to Run

**Requirements**: Python 3, and a Gemini API key (used for both embeddings and generation).

1. Clone the repo and create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip3 install langchain langchain_community langchain_classic langchain-chroma langchain-google-genai pypdf rank_bm25 chromadb spacy pandas matplotlib
python -m spacy download en_core_web_sm
```

3. Set your Gemini API key in a `.env` file at the repo root:
```
GEMINI_API_KEY=your-key-here
```

4. Open [main.ipynb](main.ipynb) and run the cells top to bottom for a full run (ingestion → chunking → embedding → indexing → hybrid retrieval → generation → evaluation).

`./my_chroma_db` (the persisted vector store) and `.env` are both gitignored — on a fresh clone you'll need to re-run the ingestion/embedding cells once to rebuild the vector store locally before the "reload persisted data" and evaluation sections will have anything to load.
