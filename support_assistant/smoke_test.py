"""Run the two required offline mock-mode graph examples."""

import json

from .main import build_graph


def main() -> None:
    graph = build_graph()
    for query in ["How long does delivery take?", "What is the capital of France?"]:
        print(json.dumps(graph.invoke({"query": query})["response"], indent=2))


if __name__ == "__main__":
    main()
