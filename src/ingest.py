from pathlib import Path
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("data/enterprise_docs")
PERSIST_DIR = "storage/enterprise_chroma_db"


def load_all_documents():
    documents = []
    for file_path in DATA_DIR.glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()

        # Extract the real doc_id from the filename: dsid_xxxx__title.txt
        doc_id = file_path.name.split("__")[0]

        for doc in docs:
            doc.metadata["doc_id"] = doc_id
            doc.metadata["title"] = file_path.stem.split("__", 1)[-1]

        documents.extend(docs)

    return documents


def split_into_chunks(documents, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks, persist_path=PERSIST_DIR):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Path("storage").mkdir(exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path
    )
    return vectorstore


if __name__ == "__main__":
    docs = load_all_documents()
    print(f"Total documents loaded: {len(docs)}")

    chunks = split_into_chunks(docs)
    print(f"Total chunks created: {len(chunks)}")
    print(f"Sample chunk metadata: {chunks[0].metadata}")

    vectorstore = build_vectorstore(chunks)
    print(f"Vector store created with {vectorstore._collection.count()} chunks")