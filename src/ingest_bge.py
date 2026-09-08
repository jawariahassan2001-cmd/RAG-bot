from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ingest import load_all_documents, split_into_chunks

PERSIST_DIR_BGE = "storage/enterprise_chroma_bge"

if __name__ == "__main__":
    docs = load_all_documents()
    print(f"Total documents loaded: {len(docs)}")

    chunks = split_into_chunks(docs)
    print(f"Total chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    Path("storage").mkdir(exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR_BGE
    )

    print(f"Vector store created with {vectorstore._collection.count()} chunks")
    print(f"Saved to: {PERSIST_DIR_BGE}")