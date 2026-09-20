import os
from database import SessionLocal
from retrieval import search
from state import State
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()


MAX_RETRIES = 2

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


class Grade(BaseModel):
    relevant: bool = Field(description="True if the chunk helps answer the question")
    reason: str = Field(description="One short sentence explaining the judgement")


grader = model.with_structured_output(Grade)

GRADE_PROMPT = """Decide whether this document helps answer the question.

Relevant means it contains information that answers the question.
A document on a related topic that does not answer the question is NOT relevant.

Question: {question}

Document:
{document}"""


def grade_node(state: State):
    kept = []
    
    for chunk, distance in state["chunks"]:
        result = grader.invoke(
            GRADE_PROMPT.format(question=state["question"], document=chunk.content[:1500])
        )
        print(f"[grade] p{chunk.page}: {result.relevant} — {result.reason}")

        if result.relevant:
            kept.append((chunk, distance))

    print(f"[grade] kept {len(kept)} of {len(state['chunks'])}")
    return {"chunks": kept}

def retrieve_node(state: State):
    db = SessionLocal()
    
    try:
        matches = search(
            db, state['question'], k=5, max_distance = 0.6
        )
    finally:
        db.close()
        
    print(f"[retrieve] {len(matches)} chunks for: {state['question']}")
    return {"chunks": matches}


def decide_after_grading(state: State) -> str:
    if state["chunks"]:
        print("[decide] have chunks -> generate")
        return "generate"

    if state["retry_count"] >= MAX_RETRIES:
        print("[decide] out of retries -> give up")
        return "give_up"

    print("[decide] nothing relevant -> rewrite")
    return "rewrite"



REWRITE_PROMPT = """This question retrieved no relevant documents from a technical reference.

Rewrite it using terminology more likely to appear in the document. Keep the same
intent. Return only the rewritten question, nothing else.

Original question: {question}"""


def rewrite_node(state: State) -> dict:
    response = model.invoke(REWRITE_PROMPT.format(question=state["question"]))
    rewritten = response.content.strip()

    print(f"[rewrite] {state['question']!r} -> {rewritten!r}")
    return {
        "question": rewritten,
        "retry_count": state["retry_count"] + 1,
    }


GENERATE_PROMPT = """Answer the question using only the context below.
Cite the page number for each claim, like (p17).

Context:
{context}

Question: {question}"""


def build_context(chunks) -> str:
    parts = []
    for chunk, distance in chunks:
        parts.append(f"[Page {chunk.page} — {chunk.heading}]\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def generate_node(state: State) -> dict:
    context = build_context(state["chunks"])
    response = model.invoke(
        GENERATE_PROMPT.format(context=context, question=state["original_question"])
    )

    print(f"[generate] {len(response.content)} chars")
    return {"generation": response.content}


def give_up_node(state: State) -> dict:
    print("[give_up] no answer")
    return {"generation": "I couldn't find anything in the document about that."}

if __name__ == "__main__":
    # result = {
    #     "question": "how do background tasks work?",
    #     "original_question": "how do background tasks work?",
    #     "chunks": [],
    #     "generation": "",
    #     "retry_count": 0,
    # }

    # result.update(retrieve_node(result))
    # result.update(grade_node(result))
    
    base = {"question": "x", "original_question": "x", "chunks": [],
        "generation": "", "retry_count": 0}

    # print(decide_after_grading({**base, "chunks": ["something"]}))
    # print(decide_after_grading({**base, "chunks": [], "retry_count": 0}))
    # print(decide_after_grading({**base, "chunks": [], "retry_count": 2}))
    print(rewrite_node({**base, "question": "what is the capital of France?"}))