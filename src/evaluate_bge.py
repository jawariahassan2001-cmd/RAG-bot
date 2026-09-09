import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

PERSIST_DIR = "storage/enterprise_chroma_bge"
QUESTIONS_FILE = "questions.jsonl"


def load_fully_answerable_questions():
    corpus_ids = set(f.split("__")[0] for f in os.listdir("data/enterprise_docs"))
    questions = []
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            expected = q.get("expected_doc_ids", [])
            if expected and set(expected).issubset(corpus_ids):
                questions.append(q)
    return questions


def precision_recall(retrieved_docs, expected_ids):
    retrieved_ids = [d.metadata.get("doc_id") for d in retrieved_docs]
    matches = [rid for rid in retrieved_ids if rid in expected_ids]
    precision = len(matches) / len(retrieved_ids) if retrieved_ids else 0
    found = set(retrieved_ids) & set(expected_ids)
    recall = len(found) / len(expected_ids) if expected_ids else 0
    return precision, recall


def average(values):
    return sum(values) / len(values) if values else 0


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

baseline_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

wide_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=wide_retriever
)

questions = load_fully_answerable_questions()
print(f"Evaluating BGE embeddings on {len(questions)} questions...\n")

baseline_p, baseline_r = [], []
reranked_p, reranked_r = [], []

for i, q in enumerate(questions, 1):
    expected = q["expected_doc_ids"]

    base_docs = baseline_retriever.invoke(q["question"])
    p, r = precision_recall(base_docs, expected)
    baseline_p.append(p)
    baseline_r.append(r)

    rerank_docs = reranking_retriever.invoke(q["question"])
    p2, r2 = precision_recall(rerank_docs, expected)
    reranked_p.append(p2)
    reranked_r.append(r2)

    if i % 10 == 0:
        print(f"Processed {i}/{len(questions)}...")

print("\n=== BGE RESULTS ===")
print(f"BGE baseline  — avg precision: {average(baseline_p):.2f}, avg recall: {average(baseline_r):.2f}")
print(f"BGE re-ranked — avg precision: {average(reranked_p):.2f}, avg recall: {average(reranked_r):.2f}")