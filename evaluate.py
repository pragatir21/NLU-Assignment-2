def load_names(filepath):
    """
    Loads names from a text file, converts them to lowercase, 
    and strips whitespace for accurate comparison.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Lowercase for case-insensitive matching
            names = [line.strip().lower() for line in f.readlines() if line.strip()]
        return names
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []

def calculate_evaluation_metrics(training_names_list, generated_names_list):
    """
    Computes the Novelty Rate and Diversity for a list of generated names.
    """
    if not generated_names_list:
        return 0.0, 0.0

    total_generated = len(generated_names_list)
    
    # Convert generated names to lowercase for accurate math
    gen_names_lower = [name.lower() for name in generated_names_list]
    
    # 1. Calculate Diversity
    unique_generated = set(gen_names_lower)
    diversity = len(unique_generated) / total_generated
    
    # 2. Calculate Novelty Rate
    training_set = set(training_names_list) # Already lowercased by load_names
    
    novel_names_count = sum(1 for name in gen_names_lower if name not in training_set)
    novelty_rate = (novel_names_count / total_generated) * 100
    
    return novelty_rate, diversity

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # 1. Load the training data
    TRAINING_DATA_PATH = "Training Names.txt"
    training_names = load_names(TRAINING_DATA_PATH)
    
    print(f"Loaded {len(training_names)} training names.\n")
    
    # 2. Model Outputs (Plugged in from the 50-epoch runs)
    rnn_generated_names = [
        "Reshma", "Dini", "Dived", "Akshay", "Naveen", 
        "Tarvi", "Anupa", "Naveek", "Param", "Nirav"
    ]
    
    blstm_generated_names = [
        "Umashankar", "Nagaraj", "Rakhi", "Rohit", "Shiva", 
        "Lakshman", "Tathagata", "Vidya", "Tulsidas", "Dhanashan"
    ]
    
    attention_generated_names = [
        "Bhakti", "Janak", "Bhoomi", "Shashika", "Shakti", 
        "Somesh", "Ramakatt", "Dharmendra", "Sovit", "Addevivi"
    ]
    
    models_to_evaluate = {
        "Vanilla RNN": rnn_generated_names,
        "BLSTM": blstm_generated_names,
        "RNN with Attention": attention_generated_names
    }
    
    # 3. Compute and display metrics for each model
    print("--- Quantitative Evaluation Results ---")
    print(f"{'Model':<25} | {'Novelty Rate (%)':<18} | {'Diversity':<10}")
    print("-" * 60)
    
    for model_name, gen_names in models_to_evaluate.items():
        novelty, diversity = calculate_evaluation_metrics(training_names, gen_names)
        print(f"{model_name:<25} | {novelty:>16.2f} % | {diversity:>9.4f}")