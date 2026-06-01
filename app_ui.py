import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Page title
st.title("RAG Customer Support Assistant")

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful customer support assistant.

Answer the question only using the provided context.

If the answer is not in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}
""")

# User input
query = st.text_input("Ask your question")

# Button
if st.button("Get Answer"):

    # Retrieve documents
    docs = retriever.invoke(query)

    # Combine context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Final prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Generate answer
    response = llm.invoke(final_prompt)

    # Display answer
    st.subheader("Answer")
    st.write(response.content)

    # Display sources
    st.subheader("Sources")

    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "Unknown")
        st.write(f"Source {i+1} → Page {page}")

    # HITL Escalation
    low_confidence_phrases = [
        "I could not find",
        "not available",
        "don't know"
    ]

    if any(phrase.lower() in response.content.lower() for phrase in low_confidence_phrases):
        st.error("Escalation Triggered: Forwarding query to human support agent...")
