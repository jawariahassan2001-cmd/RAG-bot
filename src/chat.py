import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# ==================== SETUP (runs once) ====================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="storage/chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

template = """You are a helpful assistant for NovaGuard Technologies' IT policy.

If the user greets you or makes small talk (like "hi", "hello", "thanks"), respond naturally and briefly, and ask how you can help with the IT policy.

If the user asks a real question, answer it using ONLY the context below. If the answer isn't in the context, say "I don't have that information in the policy."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    max_tokens=500
)

print("RAG chatbot ready. Ask a question about the IT policy (type 'quit' to exit).\n")

# ==================== LOOP (runs per question) ====================

while True:
    question = input("You: ")

    if question.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    results = retriever.invoke(question)

    context_text = "\n\n".join(doc.page_content for doc in results)
    formatted_prompt = prompt.invoke({"context": context_text, "question": question})
    response = llm.invoke(formatted_prompt)

    print(f"\nBot: {response.content}\n")