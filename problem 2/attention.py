import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# ==========================================
# 1. RNN with Basic Attention Implemented from Scratch
# ==========================================
class BasicAttentionScratch(nn.Module):
    """
    Computes attention weights and the context vector given the current 
    hidden state and all previous hidden states.
    """
    def __init__(self, hidden_size):
        super(BasicAttentionScratch, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        
        # Shape becomes: (seq_len, batch_size, hidden_size)
        hidden_repeated = hidden.unsqueeze(0).repeat(seq_len, 1, 1)
        
        # Calculate alignment scores
        energy = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2) # Shape: (seq_len, batch_size)
        
        # Transpose to get Shape: (batch_size, seq_len)
        attention_scores = attention_scores.t()
        
        # Apply softmax to convert scores into probabilities
        attn_weights = F.softmax(attention_scores, dim=1)
        
        # Compute the context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0, 1))
        
        return context.squeeze(1), attn_weights

class RNNAttentionScratch(nn.Module):
    """
    A Character-Level RNN that utilizes the Basic Attention mechanism.
    """
    def __init__(self, vocab_size, hidden_size):
        super(RNNAttentionScratch, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.i2h = nn.Linear(vocab_size + hidden_size, hidden_size)
        self.attention = BasicAttentionScratch(hidden_size)
        self.h2o = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, x, hidden, past_states):
        combined_input = torch.cat((x, hidden), dim=1)
        hidden = torch.tanh(self.i2h(combined_input))
        
        if past_states.size(0) > 0:
            context, attn_weights = self.attention(hidden, past_states)
        else:
            context = torch.zeros_like(hidden)
            
        combined_output = torch.cat((hidden, context), dim=1)
        output = self.h2o(combined_output)
        
        return output, hidden

    def init_hidden(self):
        """Added to initialize the hidden state at the start of a sequence."""
        return torch.zeros(1, self.hidden_size)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 2. Data Preparation
# ==========================================
def load_and_prep_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            names = [line.strip().lower() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")
        return [], [], {}, {}
        
    chars = sorted(list(set(''.join(names))))
    vocab = ['<SOS>', '<EOS>'] + chars
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}
    
    return names, vocab, char_to_idx, idx_to_char

def sequence_to_tensor(seq, char_to_idx, vocab_size):
    tensor = torch.zeros(len(seq), 1, vocab_size)
    for li, token in enumerate(seq):
        tensor[li][0][char_to_idx[token]] = 1
    return tensor

def target_to_tensor(name, char_to_idx):
    letter_indexes = [char_to_idx[letter] for letter in name]
    letter_indexes.append(char_to_idx['<EOS>'])
    return torch.LongTensor(letter_indexes)

# ==========================================
# 3. Training Loop
# ==========================================
def train_attention(model, names, char_to_idx, vocab_size, epochs=150, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nStarting Attention Model training for {epochs} epochs...")
    
    for epoch in range(epochs):
        random.shuffle(names)
        total_loss = 0
        
        for name in names:
            input_seq = ['<SOS>'] + list(name)
            input_tensor = sequence_to_tensor(input_seq, char_to_idx, vocab_size)
            target_tensor = target_to_tensor(name, char_to_idx)
            
            hidden = model.init_hidden()
            
            # Initialize empty tensor to store past hidden states
            # Shape needs to be (seq_len, batch_size, hidden_size)
            past_states = torch.empty(0, 1, model.hidden_size) 
            
            model.zero_grad()
            loss = 0
            
            for i in range(input_tensor.size(0)):
                output, hidden = model(input_tensor[i], hidden, past_states)
                loss += criterion(output, target_tensor[i].unsqueeze(0))
                
                # Append the newly calculated hidden state to our history for the next step
                past_states = torch.cat((past_states, hidden.unsqueeze(0)), dim=0)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item() / input_tensor.size(0)
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Average Loss: {total_loss / len(names):.4f}")
            
    return model

# ==========================================
# 4. Name Generation 
# ==========================================
def generate_name(model, char_to_idx, idx_to_char, vocab_size, max_length=15, temperature=0.8):
    """Generates a name using the Attention model with temperature sampling."""
    with torch.no_grad():
        input_tensor = sequence_to_tensor(['<SOS>'], char_to_idx, vocab_size)
        hidden = model.init_hidden()
        past_states = torch.empty(0, 1, model.hidden_size)
        output_name = ""
        
        for _ in range(max_length):
            output, hidden = model(input_tensor[0], hidden, past_states)
            past_states = torch.cat((past_states, hidden.unsqueeze(0)), dim=0)
            
            # --- STOCHASTIC SAMPLING ---
            output_dist = output.squeeze(0) / temperature
            probabilities = torch.softmax(output_dist, dim=0)
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
    filepath = "Training Names.txt"
    names, vocab, char_to_idx, idx_to_char = load_and_prep_data(filepath)
    
    if not names:
        exit()
        
    VOCAB_SIZE = len(vocab)
    HIDDEN_SIZE = 128
    LEARNING_RATE = 0.001
    LAYERS = 1
    EPOCHS = 50
    
    attention_model = RNNAttentionScratch(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE)
    
    print("--- RNN with Basic Attention Model Details ---")
    print(f"Hyperparameters:")
    print(f"  - Hidden Size: {HIDDEN_SIZE}")
    print(f"  - Layers: {LAYERS} (Plus Attention Module)")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"Total Trainable Parameters: {count_parameters(attention_model)}")
    
    # Train
    trained_attn = train_attention(attention_model, names, char_to_idx, VOCAB_SIZE, epochs=EPOCHS, lr=LEARNING_RATE)
    
    # Generate
    print("\n--- Generated Attention Name Samples ---")
    for _ in range(10):
        print(generate_name(trained_attn, char_to_idx, idx_to_char, VOCAB_SIZE))