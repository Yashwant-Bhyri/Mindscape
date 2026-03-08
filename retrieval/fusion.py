def rrf_fusion(results_list, k=60):
    """
    Merge multiple lists of retrieval results using Reciprocal Rank Fusion.
    Each result in the list must have an 'chunk_id' or 'text' to identify it.
    """
    fused_scores = {}
    doc_map = {} # To keep track of the original doc content
    
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc.get('chunk_id') or doc.get('text')
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
                doc_map[doc_id] = doc
            
            fused_scores[doc_id] += 1.0 / (k + rank + 1)
            
    # Sort by score
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for doc_id, score in sorted_docs:
        doc = doc_map[doc_id].copy()
        doc['rrf_score'] = score
        final_results.append(doc)
        
    return final_results

if __name__ == "__main__":
    # Test
    res1 = [{'chunk_id': 'A', 'text': 'doc a'}, {'chunk_id': 'B', 'text': 'doc b'}]
    res2 = [{'chunk_id': 'B', 'text': 'doc b'}, {'chunk_id': 'C', 'text': 'doc c'}]
    fused = rrf_fusion([res1, res2])
    for f in fused:
        print(f"[{f['rrf_score']:.4f}] {f['chunk_id']}")
