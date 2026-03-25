import os
from gensim.models import Word2Vec

def analyze_semantics(model_path):
    """
    Loads a trained Word2Vec model and performs semantic analysis using cosine similarity.
    This includes finding nearest neighbors and solving word analogies tailored to the IITJ corpus.
    """
    print(f"\nLoading model: {model_path}...")
    
    try:
        model = Word2Vec.load(model_path)
    except FileNotFoundError:
        print(f"Error: Could not find the model at {model_path}. Did you run Task 2?")
        return

    # ==========================================
    # 1. Nearest Neighbors (Cosine Similarity)
    # ==========================================
    print("\n--- 1. Top 5 Nearest Neighbors ---")
    
    target_words = ['research', 'student', 'phd', 'examination']
    
    for word in target_words:
        try:
            neighbors = model.wv.most_similar(word, topn=5)
            print(f"\nNearest neighbors for '{word}':")
            for neighbor, similarity_score in neighbors:
                print(f"  - {neighbor} (Similarity: {similarity_score:.4f})")
        except KeyError:
            print(f"\nWord '{word}' not in vocabulary. (Check your corpus or lower your min_count).")

    # ==========================================
    # 2. Analogy Experiments
    # ==========================================
    print("\n--- 2. Analogy Experiments ---")
    print("Format: Word A is to Word B as Word C is to ? (A : B :: C : ?)")
    
    # Analogies designed specifically for the IITJ text that was scraped
    analogies = [
        # Testing Degree Relationships
        {'positive': ['mtech', 'bachelors'], 'negative': ['btech'], 'label': 'BTech : Bachelors :: MTech : ? (Expected: Masters)'},
        
        # Testing CS Course Phrase Collocations (Machine Learning / Artificial Intelligence)
        {'positive': ['artificial', 'learning'], 'negative': ['machine'], 'label': 'Machine : Learning :: Artificial : ? (Expected: Intelligence)'},
        
        # Testing CS Course Phrase Collocations (Data Structures / Operating Systems)
        {'positive': ['operating', 'structures'], 'negative': ['data'], 'label': 'Data : Structures :: Operating : ? (Expected: Systems)'},
        
        # Testing Domain Shift (Academic vs Medical)
        {'positive': ['academic', 'hospital'], 'negative': ['health'], 'label': 'Health : Hospital :: Academic : ? (Expected: Institute/Senate)'}
    ]
    
    for analogy in analogies:
        print(f"\nEvaluating: {analogy['label']}")
        
        # Check if all words exist in the vocab first to prevent crashes
        missing_words = [w for w in analogy['positive'] + analogy['negative'] if w not in model.wv.key_to_index]
        
        if missing_words:
            print(f"  -> Cannot evaluate. Missing words in vocab: {missing_words}")
            continue
            
        try:
            # Calculate the analogy using vector arithmetic: positive[0] + positive[1] - negative[0]
            result = model.wv.most_similar(positive=analogy['positive'], negative=analogy['negative'], topn=1)
            predicted_word = result[0][0]
            score = result[0][1]
            print(f"  -> Predicted: {predicted_word} (Score: {score:.4f})")
        except Exception as e:
            print(f"  -> Error evaluating analogy: {e}")

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    MODEL_TO_TEST = "./trained_models/w2v_cbow_d50_w5_n10.model"    
    if os.path.exists(MODEL_TO_TEST):
        analyze_semantics(MODEL_TO_TEST)
    else:
        print(f"Please update MODEL_TO_TEST. The file '{MODEL_TO_TEST}' does not exist.")