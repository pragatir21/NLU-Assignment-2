import torch
import torch.nn as nn
import torch.optim as optim
import random

# ==========================================
# 1. Vanilla RNN Implemented from Scratch
# ==========================================
class VanillaRNN(nn.Module):
    """
    A Character-Level Vanilla Recurrent Neural Network implemented from scratch.
    It takes a one-hot encoded character and the previous hidden state to predict 
    the next character in a name.
    """
    def __init__(self, vocab_size, hidden_size):
        super(VanillaRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Transformations
        self.i2h = nn.Linear(vocab_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden):
        # Combine input and previous hidden state, then apply tanh activation
        hidden = torch.tanh(self.i2h(x) + self.h2h(hidden))
        # Generate the output prediction
        output = self.h2o(hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 2. Data Preparation
# ==========================================
def load_and_prep_data(filepath):
    """Loads names and creates character-to-index mappings."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            names = [line.strip().lower() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Make sure it's in the same folder.")
        return [], [], {}, {}
        
    # Create vocabulary (all unique characters in dataset + Start/End tokens)
    chars = sorted(list(set(''.join(names))))
    vocab = ['<SOS>', '<EOS>'] + chars
    
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}
    
    return names, vocab, char_to_idx, idx_to_char

def sequence_to_tensor(seq, char_to_idx, vocab_size):
    """Converts a sequence of tokens into a one-hot encoded PyTorch tensor."""
    tensor = torch.zeros(len(seq), 1, vocab_size)
    for li, token in enumerate(seq):
        tensor[li][0][char_to_idx[token]] = 1
    return tensor

def target_to_tensor(name, char_to_idx):
    """Converts target sequence into integer class labels."""
    letter_indexes = [char_to_idx[letter] for letter in name]
    letter_indexes.append(char_to_idx['<EOS>']) # Append End-Of-Sequence token
    return torch.LongTensor(letter_indexes)

# ==========================================
# 3. Training Loop
# ==========================================
def train_rnn(model, names, char_to_idx, vocab_size, epochs=150, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        random.shuffle(names) # Shuffle dataset each epoch
        total_loss = 0
        
        for name in names:
            # Prepare input sequence as a list of tokens: ['<SOS>', 'a', 'r', 'j', 'u', 'n']
            input_seq = ['<SOS>'] + list(name)
            input_tensor = sequence_to_tensor(input_seq, char_to_idx, vocab_size)
            target_tensor = target_to_tensor(name, char_to_idx)
            
            hidden = model.init_hidden()
            model.zero_grad()
            loss = 0
            
            # Forward pass through the sequence
            for i in range(input_tensor.size(0)):
                output, hidden = model(input_tensor[i], hidden)
                loss += criterion(output, target_tensor[i].unsqueeze(0))
                
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() / input_tensor.size(0)
            
        # Print update every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Average Loss: {total_loss / len(names):.4f}")
        
    return model

# ==========================================
# 4. Name Generation 
# ==========================================
def generate_name(model, char_to_idx, idx_to_char, vocab_size, max_length=15, temperature=0.8):
    """Generates a single new name using the trained model with temperature sampling."""
    with torch.no_grad():
        input_tensor = sequence_to_tensor(['<SOS>'], char_to_idx, vocab_size)
        hidden = model.init_hidden()
        output_name = ""
        
        for _ in range(max_length):
            output, hidden = model(input_tensor[0], hidden)
            
            # 1. Scale the logits by the temperature
            output_dist = output.squeeze(0) / temperature
            
            # 2. Convert to probabilities using Softmax
            probabilities = torch.softmax(output_dist, dim=0)
            
            # 3. Sample from the probability distribution instead of using argmax
            topi = torch.multinomial(probabilities, 1).item()
            
            next_char = idx_to_char[topi]
            
            if next_char == '<EOS>':
                break
                
            output_name += next_char
            input_tensor = sequence_to_tensor([next_char], char_to_idx, vocab_size)
            
        return output_name.capitalize()

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    filepath = "Training Names.txt"
    names, vocab, char_to_idx, idx_to_char = load_and_prep_data(filepath)
    
    if not names:
        exit()
        
    VOCAB_SIZE = len(vocab)
    HIDDEN_SIZE = 128
    LEARNING_RATE = 0.001
    LAYERS = 1
    EPOCHS = 50
    
    # 2. Initialize Model
    rnn_model = VanillaRNN(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE)
    
    print("--- Vanilla RNN Model Details ---")
    print(f"Hyperparameters:")
    print(f"  - Hidden Size: {HIDDEN_SIZE}")
    print(f"  - Layers: {LAYERS}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"Total Trainable Parameters: {count_parameters(rnn_model)}")
    
    # 3. Train Model
    trained_rnn = train_rnn(rnn_model, names, char_to_idx, VOCAB_SIZE, epochs=EPOCHS, lr=LEARNING_RATE)
    
    # 4. Generate Samples
    print("\n--- Generated Name Samples ---")
    for _ in range(10):
        print(generate_name(trained_rnn, char_to_idx, idx_to_char, VOCAB_SIZE))