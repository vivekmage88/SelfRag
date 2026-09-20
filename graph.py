from langgraph.graph import StateGraph, START, END
from state import State
from nodes import (
    retrieve_node,
    grade_node,
    decide_after_grading,
    rewrite_node,
    generate_node,
    give_up_node,
)


def build_graph():
    builder = StateGraph(State)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("generate", generate_node)
    builder.add_node("give_up", give_up_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")

    builder.add_conditional_edges(
        "grade",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "give_up": "give_up",
        },
    )

    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("give_up", END)

    return builder.compile()


def ask(question: str) -> dict:
    graph = build_graph()
    return graph.invoke({
        "question": question,
        "original_question": question,
        "chunks": [],
        "generation": "",
        "retry_count": 0,
    })


if __name__ == "__main__":
    # for q in ["how do background tasks work?", "what is the capital of France?"]:
    #     print(f"\n=== {q} ===")
    #     result = ask(q)
    #     print(f"\n{result['generation']}\n")
        
    print(build_graph().get_graph().draw_ascii())