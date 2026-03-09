import os
import time
from .corpus_loader import load_corpus
from .chunker import process_documents
from .embedding_index import DenseRetriever
from .bm25_index import SparseRetriever
from .query_generator import QueryGenerator
from .fusion import rrf_fusion

class HybridRetriever:
    def __init__(self, data_dir="data/corpus"):
        self.data_dir = data_dir
        # self.dense = DenseRetriever() # DISABLED FOR MVP OOM
        self.sparse = SparseRetriever()
        self.query_gen = QueryGenerator()
        self.is_initialized = False

    def initialize(self, force=False):
        """
        Load corpus, chunk, and build indices.
        """
        if not force and self.sparse.load_index():
            self.is_initialized = True
            return

        print("Initializing Hybrid Retrieval System...")
        docs = load_corpus(self.data_dir)
        if not docs:
            print(f"Warning: No documents found in {self.data_dir}")
            return
            
        chunks = process_documents(docs)
        # self.dense.build_index(chunks) # DISABLED FOR MVP OOM
        self.sparse.build_index(chunks)
        self.is_initialized = True
        print(f"Initialization complete. Indexed {len(chunks)} chunks.")

    def retrieve_evidence(self, transcript, bsv, top_k=5):
        """
        Multi-query hybrid retrieval with RRF fusion.
        """
        if not self.is_initialized:
            self.initialize()

        start_time = time.time()
        
        # 1. Generate Queries
        queries = self.query_gen.generate_queries(transcript, bsv)
        print(f"Retriever: Generated {len(queries)} queries.")
        
        # 2. Collect candidates
        all_query_results = []
        for q in queries:
            # dense_res = self.dense.search(q, top_k=10)
            sparse_res = self.sparse.search(q, top_k=10)
            # all_query_results.append(dense_res)
            all_query_results.append(sparse_res)
            
        # 3. Fuse (Now just flat sorting since it's only BM25)
        fused_results = rrf_fusion(all_query_results)
        
        # 4. Filter and Format
        final_evidence = fused_results[:top_k]
        
        latency = (time.time() - start_time) * 1000
        print(f"Retrieval latency: {latency:.2f}ms")
        
        return {
            "queries": queries,
            "evidence": final_evidence,
            "latency_ms": latency
        }

if __name__ == "__main__":
    # Test (Relative paths need fix or run from parent)
    # This is intended to be used as a module
    pass
