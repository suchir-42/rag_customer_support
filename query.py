from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# EMBEDDING MODEL
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# LOAD VECTOR DATABASE
# -----------------------------
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# -----------------------------
# RETRIEVER
# -----------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)

# -----------------------------
# PROMPT TEMPLATE
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are a helpful customer support assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
""")

# -----------------------------
# CHAT HISTORY
# -----------------------------
chat_history = []

# -----------------------------
# USER QUERY LOOP
# -----------------------------
while True:

    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve documents
    docs = retriever.invoke(query)

    # Combine context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Create final prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Generate response
    response = llm.invoke(final_prompt)

    # Store chat history
    chat_history.append({
        "question": query,
        "answer": response.content
    })

    # Print answer
    print("\nAnswer:\n")
    print(response.content)

    print("\nSources:\n")

    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "Unknown")
        print(f"Source {i+1} → Page {page}")

    # HITL Escalation
    low_confidence_phrases = [
        "I could not find",
        "not available",
        "don't know"
    ]

    if any(
        phrase.lower() in response.content.lower()
        for phrase in low_confidence_phrases
    ):
        print("\nEscalation Triggered:")
        print("Forwarding query to human support agent...")

    # Show retrieved chunks
    print("\n" + "=" * 60)
    print("Retrieved Context Chunks")
    print("=" * 60)

    for i, doc in enumerate(docs, start=1):
        print(f"\nChunk {i}:\n")
        print(doc.page_content[:500])
        print("\n" + "-" * 50)
