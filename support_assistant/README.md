# Support Assistant

## Run locally

From the repository root:

```powershell
python -m support_assistant.retrieval
$env:MOCK_LLM = "1"
uvicorn support_assistant.main:app --reload
```

The default mock mode is deterministic and does not call an LLM. Test it with:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" -Body '{"query":"How long does delivery take?"}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" -Body '{"query":"What is the capital of France?"}'
```

Expected mock responses have this shape:

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials...","sources":["doc_01_delivery_policy"],"confidence":1.0}
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## Architecture

Ingestion is handled by `retrieval.load_documents`, which reads the eight exact policy files and creates one chunk per document. `retrieval.embed_texts` uses the local `all-MiniLM-L6-v2` SentenceTransformer and `build_index` stores those vectors and metadata in the persistent ChromaDB collection `zepto_policy`. The LangGraph `classify_intent` node routes policy questions to `retrieve_and_answer`; that node queries ChromaDB for the top three cosine-similar chunks and generates the grounded response. General questions route to `direct_answer` without retrieval. The final response is validated by the `AnswerResponse` Pydantic model and exposed through FastAPI `POST /ask`.

Only generation/classification branches on `MOCK_LLM`: unset or `MOCK_LLM=1` uses the keyword classifier, canned grounded answer, and fixed general answer with no LLM call. `MOCK_LLM=0` uses Groq for classification and structured answer generation, with up to two corrective retries when JSON validation fails. Embedding and Chroma retrieval run in both modes.

The index uses cached `all-MiniLM-L6-v2` weights when available. To permit the first model download explicitly, set `ALLOW_MODEL_DOWNLOAD=1`; otherwise the default mock path uses deterministic local vectors and never waits on an embedding service.

## Docker

```powershell
docker build -f support_assistant/Dockerfile -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
```

Then post JSON to `http://127.0.0.1:7860/ask`.
