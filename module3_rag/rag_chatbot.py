"""Ask questions about local policy documents with retrieval and optional Groq generation."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = Path(__file__).resolve().parent / "docs"


@dataclass
class Chunk:
    source: str
    text: str


def load_chunks() -> list[Chunk]:
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        chunks.extend(Chunk(path.name, paragraph) for paragraph in paragraphs)
    return chunks


def retrieve(question: str, chunks: list[Chunk], top_k: int = 3) -> list[Chunk]:
    documents = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    indexes = scores.argsort()[::-1][:top_k]
    return [chunks[index] for index in indexes if scores[index] > 0]


def answer(question: str, chunks: list[Chunk]) -> str:
    relevant = retrieve(question, chunks)
    if not relevant:
        return "I could not find that in the policy documents. Please contact support for help."
    context = "\n\n".join(f"[{chunk.source}] {chunk.text}" for chunk in relevant)
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages=[
                    {"role": "system", "content": "Answer only from the supplied policy context. If the answer is missing, say so. Cite the source filename."},
                    {"role": "user", "content": f"Policy context:\n{context}\n\nQuestion: {question}"},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as error:
            print(f"Groq was unavailable ({error}); using extractive local mode.")
    return "Relevant policy text:\n" + "\n".join(f"- {chunk.text} ({chunk.source})" for chunk in relevant)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-samples", action="store_true", help="Skip the sample questions")
    args = parser.parse_args()
    chunks = load_chunks()
    if not chunks:
        raise SystemExit(f"No policy documents found in {DOCS_DIR}")
    print("RAG policy assistant. Type 'quit' to exit.")
    if os.getenv("GROQ_API_KEY"):
        print("Groq generation is enabled.")
    else:
        print("No GROQ_API_KEY found; local extractive answers are enabled.")
    if not args.no_samples:
        for question in ["How long do I have to return a book?", "How long does standard shipping take?"]:
            print(f"\nQ: {question}\nA: {answer(question, chunks)}")
    while True:
        try:
            question = input("\nAsk a policy question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit"}:
            break
        if question:
            print(f"\n{answer(question, chunks)}")


if __name__ == "__main__":
    main()
