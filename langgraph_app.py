from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict

from langgraph.graph import StateGraph, END

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# EMBEDDINGS
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# VECTOR DB
# -----------------------------
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)

# -----------------------------
# PROMPT
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are a helpful customer support assistant.

Answer ONLY using the provided context.

If the answer is not available in the context,
say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
""")

# -----------------------------
# STATE
# -----------------------------
class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    escalate: bool

# -----------------------------
# RETRIEVE NODE
# -----------------------------
def retrieve_node(state):

    question = state["question"]

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return {
        "question": question,
        "context": context
    }

# -----------------------------
# DECISION NODE
# -----------------------------
def decision_node(state):

    context = state["context"]
    question = state["question"]

    # Very weak retrieval
    if len(context.strip()) < 100:
        return {"escalate": True}

    # If retrieved context does not contain useful overlap
    question_words = question.lower().split()

    match_count = 0

    for word in question_words:
        if word in context.lower():
            match_count += 1

    # Low relevance → escalate
    if match_count <= 1:
        return {"escalate": True}

    return {"escalate": False}

# -----------------------------
# GENERATE NODE
# -----------------------------
def generate_node(state):

    final_prompt = prompt.invoke({
        "context": state["context"],
        "question": state["question"]
    })

    response = llm.invoke(final_prompt)

    return {
        "answer": response.content
    }

# -----------------------------
# ESCALATION NODE
# -----------------------------
def escalation_node(state):

    return {
        "answer": (
            "This question requires human support escalation."
        )
    }

# -----------------------------
# BUILD GRAPH
# -----------------------------
graph = StateGraph(GraphState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("decision", decision_node)
graph.add_node("generate", generate_node)
graph.add_node("escalate", escalation_node)

# Flow
graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "decision")

graph.add_conditional_edges(
    "decision",
    lambda state: "escalate"
    if state["escalate"]
    else "generate",
    {
        "generate": "generate",
        "escalate": "escalate"
    }
)

graph.add_edge("generate", END)
graph.add_edge("escalate", END)

# Compile graph
app = graph.compile()

# -----------------------------
# CHAT LOOP
# -----------------------------
while True:

    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    result = app.invoke({
        "question": question
    })

    print("\nAnswer:\n")
    print(result["answer"])
