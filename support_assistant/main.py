"""LangGraph + ChromaDB + FastAPI policy assistant with a deterministic mock baseline."""

from __future__ import annotations

import json
import os
from typing import TypedDict

from fastapi import FastAPI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

try:
    from .retrieval import build_index, collection, retrieve
except ImportError:
    from retrieval import build_index, collection, retrieve

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"
POLICY_KEYWORDS = ("delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours")

STRUCTURED_PROMPT = """ROLE: You are Zepto's policy support assistant.
CONTEXT: Use only the retrieved Zepto policy chunks below.
TASK: Answer the user's question and cite the chunk IDs that support the answer.
FORMAT: Return valid JSON with exactly answer (string), sources (list of IDs), and confidence (number from 0 to 1).
LENGTH: Keep the answer concise, no more than 80 words.
NEGATIVE CONSTRAINT: Do not answer using information that is not present in the provided context; say that the policy does not specify it.
FEW-SHOT EXAMPLE:
Question: How long does standard delivery take?
Context: delivery policy says 10 to 30 minutes.
JSON: {"answer":"Standard delivery takes 10 to 30 minutes.","sources":["doc_01_delivery_policy"],"confidence":1.0}
"""


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class GraphState(TypedDict, total=False):
    query: str
    intent: str
    context: list[dict]
    response: dict


def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]
    if MOCK_LLM:
        intent = "policy_question" if any(keyword in query.lower() for keyword in POLICY_KEYWORDS) else "general_question"
    else:
        intent = real_llm_classify(query)
    return {**state, "intent": intent}


def retrieve_and_answer(state: GraphState) -> GraphState:
    context = retrieve(state["query"], top_k=3)
    if MOCK_LLM:
        top_snippet = context[0]["text"][:200] if context else "No matching policy context was found."
        response = {"answer": f"Based on the retrieved context: {top_snippet}", "sources": [item["id"] for item in context], "confidence": 1.0}
    else:
        response = real_llm_answer(state["query"], context)
    return {**state, "context": context, "response": AnswerResponse.model_validate(response).model_dump()}


def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM:
        response = {"answer": "I can only answer questions about Zepto policies right now.", "sources": [], "confidence": 1.0}
    else:
        response = real_llm_answer(state["query"], [])
    return {**state, "response": AnswerResponse.model_validate(response).model_dump()}


def route_after_classify(state: GraphState) -> str:
    return state["intent"]


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_and_answer", retrieve_and_answer)
    builder.add_node("direct_answer", direct_answer)
    builder.set_entry_point("classify_intent")
    builder.add_conditional_edges("classify_intent", route_after_classify, {"policy_question": "retrieve_and_answer", "general_question": "direct_answer"})
    builder.add_edge("retrieve_and_answer", END)
    builder.add_edge("direct_answer", END)
    return builder.compile()


def real_client():
    from groq import Groq
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def real_llm_classify(query: str) -> str:
    result = real_client().chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[{"role": "system", "content": "Return exactly policy_question or general_question."}, {"role": "user", "content": query}],
        temperature=0,
    )
    value = result.choices[0].message.content.strip()
    return "policy_question" if "policy_question" in value else "general_question"


def real_llm_answer(query: str, context: list[dict]) -> dict:
    context_text = "\n".join(f"{item['id']}: {item['text']}" for item in context)
    messages = [{"role": "system", "content": STRUCTURED_PROMPT}, {"role": "user", "content": f"Question: {query}\nContext:\n{context_text}"}]
    last_error = "unknown validation error"
    for attempt in range(3):
        if attempt:
            messages.append({"role": "user", "content": f"Corrective instruction: return only valid JSON matching answer, sources, confidence. Previous validation error: {last_error}"})
        raw = real_client().chat.completions.create(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), messages=messages, temperature=0).choices[0].message.content
        try:
            parsed = json.loads(raw)
            return AnswerResponse.model_validate(parsed).model_dump()
        except (json.JSONDecodeError, ValidationError) as error:
            last_error = str(error)
    return {"answer": "The real-LLM response could not be validated.", "sources": [], "confidence": 0.0}


app = FastAPI(title="Zepto Policy Support Assistant")


@app.on_event("startup")
def startup() -> None:
    if collection().count() < 8:
        build_index()


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QueryRequest) -> AnswerResponse:
    result = build_graph().invoke({"query": request.query})
    return AnswerResponse.model_validate(result["response"])


if __name__ == "__main__":
    build_index()
    graph = build_graph()
    for question in ["How long does delivery take?", "What is the capital of France?"]:
        print(json.dumps(graph.invoke({"query": question})["response"], indent=2))
