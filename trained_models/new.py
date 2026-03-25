from gensim.models import Word2Vec

# Load your 300D model (ensure you trained one with size=300)
model = Word2Vec.load("w2v_cbow_d300_w5_n10.model")
word = "algorithm"
vector = model.wv[word]

# Format as a comma-separated list
vector_str = ", ".join([f"{val:.4f}" for val in vector])
print(f"{word} - {vector_str}")