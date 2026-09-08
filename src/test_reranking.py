import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

PERSIST_DIR = "storage/enterprise_chroma_db"
QUESTIONS_FILE = "questions.jsonl"

def load_fully_answerable_questions():
    corpus_ids = set()
    for fname in os.listdir("data/enterprise_docs"):
        corpus_ids.add(fname.split("__")[0])

    questions = []
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            expected = q.get("expected_doc_ids", [])
            if expected and set(expected).issubset(corpus_ids):
                questions.append(q)
    return questions

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

questions = load_fully_answerable_questions()
sample = questions[0]

print(f"Question: {sample['question']}")
print(f"Expected doc_id(s): {sample['expected_doc_ids']}\n")

results = reranking_retriever.invoke(sample["question"])

print(f"Retrieved {len(results)} chunks AFTER re-ranking:\n")
for i, doc in enumerate(results, 1):
    retrieved_id = doc.metadata.get("doc_id")
    is_match = retrieved_id in sample["expected_doc_ids"]
    marker = "✅ MATCH" if is_match else "❌ no match"
    print(f"{i}. [{marker}] doc_id={retrieved_id}")
    print(f"   {doc.page_content[:150]}...\n")