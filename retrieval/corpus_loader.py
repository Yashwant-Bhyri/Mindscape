import os
import json

def load_corpus(data_dir):
    """
    Recursively load all JSON files from the data_dir.
    Returns a list of document dictionaries.
    """
    documents = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        doc = json.load(f)
                        documents.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return documents

if __name__ == "__main__":
    # Test
    docs = load_corpus("data/corpus")
    print(f"Loaded {len(docs)} documents.")
    for d in docs:
        print(f"- {d['title']} ({d['source']})")
