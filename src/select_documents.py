import json
import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("confluence_data/confluence")
OUTPUT_DIR = Path("data/enterprise_docs")
QUESTIONS_FILE = "questions.jsonl"
TARGET_TOTAL = 500
RANDOM_SEED = 42

def get_referenced_doc_ids():
    """Find every confluence doc_id mentioned in expected_doc_ids."""
    referenced = set()
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            referenced.update(q.get("expected_doc_ids", []))
    return referenced

def main():
    all_files = list(SOURCE_DIR.glob("*.txt"))
    doc_id_to_file = {f.name.split("__")[0]: f for f in all_files}

    referenced_ids = get_referenced_doc_ids()
    # Only keep ones that actually exist in this slice
    referenced_in_slice = referenced_ids & doc_id_to_file.keys()

    remaining_ids = list(doc_id_to_file.keys() - referenced_in_slice)
    random.seed(RANDOM_SEED)
    random.shuffle(remaining_ids)

    needed_random = TARGET_TOTAL - len(referenced_in_slice)
    random_fill_ids = remaining_ids[:needed_random]

    selected_ids = referenced_in_slice | set(random_fill_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for doc_id in selected_ids:
        src_file = doc_id_to_file[doc_id]
        shutil.copy(src_file, OUTPUT_DIR / src_file.name)

    print(f"Referenced documents found in slice: {len(referenced_in_slice)}")
    print(f"Random distractor documents added: {len(random_fill_ids)}")
    print(f"Total documents copied: {len(selected_ids)}")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()