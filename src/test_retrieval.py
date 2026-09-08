import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "storage/enterprise_chroma_db"
QUESTIONS_FILE = "questions.jsonl"

import os

def load_fully_answerable_questions():
    # Get the doc_ids actually present in our corpus
    corpus_ids = set()
    for fname in os.listdir("data/enterprise_docs"):
        corpus_ids.add(fname.split("__")[0])

    questions = []
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            expected = q.get("expected_doc_ids", [])
            # Only keep questions where ALL expected docs are in our corpus
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

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

questions = load_fully_answerable_questions()
sample = questions[0]

print(f"Question: {sample['question']}")
print(f"Expected doc_id(s): {sample['expected_doc_ids']}")
print(f"Gold answer: {sample['gold_answer']}\n")

results = retriever.invoke(sample["question"])

print(f"Retrieved {len(results)} chunks:\n")
for i, doc in enumerate(results, 1):
    retrieved_id = doc.metadata.get("doc_id")
    is_match = retrieved_id in sample["expected_doc_ids"]
    marker = "✅ MATCH" if is_match else "❌ no match"
    print(f"{i}. [{marker}] doc_id={retrieved_id}")
    print(f"   {doc.page_content[:150]}...\n")