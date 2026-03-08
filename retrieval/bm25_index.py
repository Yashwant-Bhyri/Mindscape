import os
import json
from rank_bm25 import BM25Okapi

class SparseRetriever:
    def __init__(self, metadata_path='retrieval/chunk_metadata.json'):
        self.metadata_path = metadata_path
        self.bm25 = None
        self.chunks = []

    def build_index(self, chunks):
        """
        Tokenize chunks and build BM25 index.
        """
        self.chunks = chunks
        tokenized_corpus = [chunk['text'].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        # Note: BM25 index is usually small enough to rebuild on load for prototype,
        # but in production, you'd serialize it.

    def load_index(self, chunks_override=None):
        if chunks_override:
            self.build_index(chunks_override)
            return True
            
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            self.build_index(self.chunks)
            return True
        return False

    def search(self, query, top_k=10):
        if self.bm25 is None:
            if not self.load_index():
                return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_n:
            if scores[idx] > 0:
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(scores[idx])
                results.append(chunk)
        return results

if __name__ == "__main__":
    from corpus_loader import load_corpus
    from chunker import process_documents
    
    loader = load_corpus("data/corpus")
    chunks = process_documents(loader)
    
    retriever = SparseRetriever()
    retriever.build_index(chunks)
    
    res = retriever.search("mania symptoms", top_k=3)
    print(f"\nFound {len(res)} results:")
    for r in res:
        print(f"[{r['score']:.4f}] {r['title']} - {r['text'][:50]}...")
