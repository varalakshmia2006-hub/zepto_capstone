FROM python:3.12-slim

WORKDIR /app
ENV MOCK_LLM=1 \
	HF_HOME=/opt/huggingface \
	SENTENCE_TRANSFORMERS_HOME=/opt/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY support_assistant ./support_assistant

# Bake the embedding model and policy index into the image so startup is local.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -m support_assistant.retrieval

EXPOSE 7860
CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "7860"]
