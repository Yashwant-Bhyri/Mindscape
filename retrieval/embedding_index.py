import os
import faiss
import numpy as np
import json
import torch
from sentence_transformers import SentenceTransformer

# PyTorch tries to use all CPU cores for computation by default. 
# Inside a web server thread (like Mesop), this causes an immediate deadlock during model.encode()
torch.set_num_threads(1)

class DenseRetriever:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', index_path='retrieval/vector_index.faiss', metadata_path='retrieval/chunk_metadata.json'):
        self.model_name = model_name
        self._model = None
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.chunks = []

    @property
    def model(self):
        if self._model is None:
            print(f"Loading SentenceTransformer: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build_index(self, chunks):
        """
        Build and save the FAISS index from a list of chunks.
        """
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Save
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f)
        print(f"Index built and saved to {self.index_path}")

    def load_index(self):
        """
        Load the FAISS index and metadata.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            return True
        return False

    def search(self, query, top_k=10):
        """
        Perform dense semantic search.
        """
        if self.index is None:
            if not self.load_index():
                return []
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(1 / (1 + distances[0][i])) # Convert L2 to similarity
                results.append(chunk)
        return results

if __name__ == "__main__":
    from corpus_loader import load_corpus
    from chunker import process_documents
    
    # Simple build and search test
    loader = load_corpus("data/corpus")
    chunks = process_documents(loader)
    
    retriever = DenseRetriever()
    retriever.build_index(chunks)
    
    res = retriever.search("I feel very sad and have no energy", top_k=3)
    print(f"\nFound {len(res)} results:")
    for r in res:
        print(f"[{r['score']:.4f}] {r['title']} - {r['text'][:50]}...")
