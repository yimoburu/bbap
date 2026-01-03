import os
import numpy as np
from scipy.spatial.distance import cosine
import config
import utils

def run_maintenance():
    """
    Scans the Vector DB for duplicate identities.
    If Person_A and Person_B are > 85% similar, suggests a merge.
    """
    print("\n🧹 Running Identity Maintenance (Clustering)...")
    speakers = utils.load_known_speakers()
    names = list(speakers.keys())
    
    suggestions = []
    
    # Compare every vector against every other vector (O(N^2))
    # Since N (speakers) is usually small (<100), this is fast.
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a = names[i]
            name_b = names[j]
            
            vec_a = speakers[name_a]
            vec_b = speakers[name_b]
            
            # Calculate Similarity (1 - Cosine Distance)
            sim = 1 - cosine(vec_a, vec_b)
            
            if sim > config.CLUSTER_MERGE_THRESHOLD:
                suggestions.append((name_a, name_b, sim))

    if suggestions:
        print(f"⚠️ FOUND {len(suggestions)} POTENTIAL DUPLICATES:")
        print("   Consider deleting the newer file or merging them:")
        for a, b, score in suggestions:
            print(f"   🔹 {a} <--> {b} (Similarity: {score:.2f})")
    else:
        print("   ✅ Vector DB is clean (No high-confidence duplicates found).")