import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

def compare_embeddings(cbow_path, sg_path, words_to_plot, method='tsne'):
    """
    Loads both CBOW and Skip-gram models, reduces dimensionality, 
    and plots them side-by-side for comparison.
    """
    print(f"Loading CBOW model: {cbow_path}...")
    print(f"Loading Skip-gram model: {sg_path}...")
    
    try:
        cbow_model = Word2Vec.load(cbow_path)
        sg_model = Word2Vec.load(sg_path)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    # Extract words that exist in BOTH vocabularies
    valid_words = []
    cbow_vectors = []
    sg_vectors = []
    
    for word in words_to_plot:
        if word in cbow_model.wv and word in sg_model.wv:
            valid_words.append(word)
            cbow_vectors.append(cbow_model.wv[word])
            sg_vectors.append(sg_model.wv[word])
        else:
            print(f"Warning: Word '{word}' missing from one or both models.")
            
    if not valid_words:
        print("Error: None of the provided words are in the vocabularies.")
        return

    # Convert to numpy arrays
    X_cbow = np.array(cbow_vectors)
    X_sg = np.array(sg_vectors)

    # Setup Dimensionality Reduction
    if method == 'pca':
        print("Applying PCA...")
        reducer = PCA(n_components=2, random_state=42)
    elif method == 'tsne':
        print("Applying t-SNE...")
        perplexity_value = min(5, len(valid_words) - 1) 
        reducer = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    else:
        print("Invalid method. Choose 'pca' or 'tsne'.")
        return

    # Transform vectors
    cbow_result = reducer.fit_transform(X_cbow)
    sg_result = reducer.fit_transform(X_sg)

    # ==========================================
    # Plotting Side-by-Side
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Word Embeddings Comparison ({method.upper()})", fontsize=16, fontweight='bold')

    # Plot 1: CBOW
    ax1.scatter(cbow_result[:, 0], cbow_result[:, 1], edgecolors='k', c='skyblue', s=80)
    for i, word in enumerate(valid_words):
        ax1.annotate(word, (cbow_result[i, 0], cbow_result[i, 1]), 
                     xytext=(5, 2), textcoords='offset points', fontsize=10)
    ax1.set_title("CBOW Model", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Skip-gram
    ax2.scatter(sg_result[:, 0], sg_result[:, 1], edgecolors='k', c='lightcoral', s=80)
    for i, word in enumerate(valid_words):
        ax2.annotate(word, (sg_result[i, 0], sg_result[i, 1]), 
                     xytext=(5, 2), textcoords='offset points', fontsize=10)
    ax2.set_title("Skip-gram Model", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to fit the main title
    plt.show()

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    target_words = [
        # Academic terms
        'research', 'student', 'phd', 'examination', 'thesis', 'study', 'course',
        # Degrees
        'btech', 'mtech', 'ug', 'pg', 'bachelor', 'master',
        # Faculty/Admin
        'faculty', 'professor', 'director', 'staff', 'department',
        # Tech/Engineering related
        'computer', 'science', 'engineering', 'electrical', 'mechanical',
        # Health Center related
        'health', 'medical', 'hospital', 'treatment', 'doctor'
    ]
    
    # Pointing to the 100-dimension, window=5, negative=10 models for a direct 1-to-1 comparison
    CBOW_PATH = "./trained_models/w2v_cbow_d100_w5_n10.model" 
    SG_PATH = "./trained_models/w2v_skip-gram_d100_w5_n10.model" 
    
    if os.path.exists(CBOW_PATH) and os.path.exists(SG_PATH):
        # 1. Generate the PCA Visualization
        compare_embeddings(CBOW_PATH, SG_PATH, target_words, method='pca')
        
        # 2. Generate the t-SNE Visualization
        compare_embeddings(CBOW_PATH, SG_PATH, target_words, method='tsne')
    else:
        print("Please ensure both CBOW and Skip-gram models exist at the specified paths.")