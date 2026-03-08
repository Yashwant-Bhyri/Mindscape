import re

def chunk_text(text, chunk_size=400, overlap=50):
    """
    Simplistic word-based chunking. 
    Returns a list of text chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def process_documents(documents, chunk_size=400, overlap=50):
    """
    Convert a list of documents into a list of chunks with metadata.
    """
    all_chunks = []
    for doc in documents:
        text = doc.get("content", "")
        text_chunks = chunk_text(text, chunk_size, overlap)
        
        for i, chunk_text_content in enumerate(text_chunks):
            chunk = {
                "chunk_id": f"{doc.get('source', 'UNK')}_{doc.get('condition', 'UNK')}_{i}".replace(" ", "_"),
                "source": doc.get("source", "Unknown"),
                "condition": doc.get("condition", "Unknown"),
                "text": chunk_text_content,
                "title": doc.get("title", "Unknown")
            }
            all_chunks.append(chunk)
    return all_chunks

if __name__ == "__main__":
    from corpus_loader import load_corpus
    docs = load_corpus("data/corpus")
    chunks = process_documents(docs)
    print(f"Generated {len(chunks)} chunks.")
    if chunks:
        print(f"Sample Chunk 0: {chunks[0]['chunk_id']}")
        print(f"Content: {chunks[0]['text'][:100]}...")
