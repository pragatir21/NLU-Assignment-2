import os
import itertools
from gensim.models import Word2Vec

# ==========================================
# Helper Function to Load Data as Sentences
# ==========================================
def load_corpus_as_sentences(data_directory):
    """
    Reads the cleaned text files and splits them into sentences, 
    then tokenizes each sentence. Word2Vec needs a list of lists 
    (list of tokenized sentences) to respect context boundaries.
    
    Args:
        data_directory (str): Path to your cleaned .txt files.
        
    Returns:
        list: A list where each element is a list of word tokens.
    """
    sentences = []
    # For simplicity, we assume the files are already cleaned and just need basic splitting by periods or newlines.
    
    for filename in os.listdir(data_directory):
        if filename.endswith(".txt"):
            with open(os.path.join(data_directory, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                # Simple split by newline 
                raw_sentences = text.split('\n')
                for sentence in raw_sentences:
                    tokens = sentence.strip().split()
                    if len(tokens) > 0:
                        sentences.append(tokens)
    return sentences

# ==========================================
# Task 2: Model Training and Experimentation
# ==========================================
def train_word2vec_models(sentences):
    """
    Trains CBOW and Skip-gram models across a grid of hyperparameters.
    Saves the models to disk for later use in Task 3 and Task 4.
    """
    # Define the hyperparameter grids to experiment with
    embedding_dimensions = [50, 100]       # (i) Embedding dimension
    context_windows = [3, 5]               # (ii) Context window size
    negative_samples = [5, 10]             # (iii) Number of negative samples
    
    # sg=0 means CBOW, sg=1 means Skip-gram
    architectures = {'CBOW': 0, 'Skip-gram': 1} 
    
    # Create a directory to save the trained models
    model_dir = "./trained_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Starting Model Training Experiments...\n")

    # Iterate through all combinations of hyperparameters
    for arch_name, sg_val in architectures.items():
        for dim, win, neg in itertools.product(embedding_dimensions, context_windows, negative_samples):
            
            print(f"Training {arch_name} | Dim: {dim} | Window: {win} | Neg Samples: {neg}")
            
            # Initialize and train the Word2Vec model
            model = Word2Vec(
                sentences=sentences,
                vector_size=dim,      # Embedding dimension
                window=win,           # Context window size
                negative=neg,         # Number of negative samples
                sg=sg_val,            # 0 for CBOW, 1 for Skip-gram
                min_count=2,          # Ignore words with total frequency lower than this
                workers=4,            # Number of CPU threads to use
                seed=42,              # Random seed for reproducibility
                epochs=150
            )
            
            # Create a descriptive filename for the model
            model_filename = f"w2v_{arch_name.lower()}_d{dim}_w{win}_n{neg}.model"
            model_path = os.path.join(model_dir, model_filename)
            
            # Save the model for future tasks
            model.save(model_path)
            
    print("\nAll models trained and saved successfully in the './trained_models' directory.")

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    DATA_DIR = "./iitj_clean_data" 
    
    if os.path.exists(DATA_DIR):
        print("Loading corpus...")
        corpus_sentences = load_corpus_as_sentences(DATA_DIR)
        
        if len(corpus_sentences) > 0:
            train_word2vec_models(corpus_sentences)
        else:
            print("Error: No sentences found. Please check your data directory.")
    else:
        print(f"Error: Directory '{DATA_DIR}' not found.")