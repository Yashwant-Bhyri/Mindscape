import sys
import os

# Fix for broken TensorFlow on system
os.environ["USE_TF"] = "0"

# Add current dir to path to allow absolute imports in submodules if needed
sys.path.append(os.getcwd())

from retrieval.hybrid_retriever import HybridRetriever

def main():
    retriever = HybridRetriever()
    retriever.initialize(force=True)
    
    # Simple test
    test_transcript = "I haven't slept and I feel like a god."
    test_bsv = {"valence": 0.9, "arousal": 0.9, "dominance": 0.9}
    
    print("\nTesting Retrieval...")
    results = retriever.retrieve_evidence(test_transcript, test_bsv)
    
    print("\nFINAL EVIDENCE:")
    for i, ev in enumerate(results['evidence']):
        print(f"{i+1}. [{ev.get('rrf_score', 0):.4f}] {ev['title']} ({ev['source']})")
        print(f"   {ev['text'][:200]}...\n")

if __name__ == "__main__":
    main()
