import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data packages (only runs the first time)
nltk.download('punkt')
nltk.download('stopwords')
# In newer versions of NLTK, 'punkt_tab' is sometimes required
try:
    nltk.download('punkt_tab')
except:
    pass

# ==========================================
# Configuration
# ==========================================
INPUT_DIR = "./iitj_data"
OUTPUT_DIR = "./iitj_clean_data"

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Set up English stop words
stop_words = set(stopwords.words('english'))

# ==========================================
# Preprocessing Function
# ==========================================
def preprocess_text(text):
    """
    Cleans raw text by removing newlines, converting to lowercase, 
    removing punctuation, tokenizing, and removing stop words.
    """
    # 1. Remove all weird PDF line breaks and extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Remove punctuation and special characters (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Tokenization (split text into individual words)
    tokens = word_tokenize(text)
    
    # 5. Remove Stop Words
    clean_tokens = [word for word in tokens if word not in stop_words]
    
    # Rejoin the clean tokens back into a single string
    clean_text = ' '.join(clean_tokens)
    
    return clean_text

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    print("Starting text preprocessing...")
    
    # Check if we have files to process
    if not os.path.exists(INPUT_DIR) or not os.listdir(INPUT_DIR):
        print(f"Error: No files found in '{INPUT_DIR}'. Please run scrape.py first.")
        exit()

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, f"cleaned_{filename}")
            
            print(f"Processing: {filename}...")
            
            # Read the raw, messy text
            with open(input_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                
            # Clean it up
            cleaned_text = preprocess_text(raw_text)
            
            # Save the clean text to the new folder
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
                
    print(f"\nPreprocessing complete! Clean files saved to '{OUTPUT_DIR}'.")