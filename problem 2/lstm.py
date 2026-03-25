import torch
import torch.nn as nn
import torch.optim as optim
import random

# ==========================================
# 1. LSTM Cell Implemented from Scratch
# ==========================================
class LSTMCellScratch(nn.Module):
    """A single LSTM cell implemented using fundamental linear transformations."""
    def __init__(self, input_size, hidden_size):
        super(LSTMCellScratch, self).__init__()
        self.hidden_size = hidden_size
        
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        
    def forward(self, x, hidden_state):
        hx, cx = hidden_state
        gates = self.x2h(x) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return hy, cy

# ==========================================
# 2. BLSTM Implemented from Scratch
# ==========================================
class BLSTMScratch(nn.Module):
    """
    A Character-Level Bidirectional LSTM. Processes the sequence forward and backward,
    then concatenates the hidden states to predict the next character.
    """
    def __init__(self, vocab_size, hidden_size):
        super(BLSTMScratch, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.lstm_forward = LSTMCellScratch(vocab_size, hidden_size)
        self.lstm_backward = LSTMCellScratch(vocab_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, x_seq):
        seq_len, batch_size, _ = x_seq.size()
        
        h_f = torch.zeros(batch_size, self.hidden_size).to(x_seq.device)
        c_f = torch.zeros(batch_size, self.hidden_size).to(x_seq.device)
        h_b = torch.zeros(batch_size, self.hidden_size).to(x_seq.device)
        c_b = torch.zeros(batch_size, self.hidden_size).to(x_seq.device)
        
        forward_states = []
        backward_states = []
        
        # Forward Pass
        for t in range(seq_len):
            x_t = x_seq[t]
            h_f, c_f = self.lstm_forward(x_t, (h_f, c_f))
            forward_states.append(h_f)
            
        # Backward Pass
        for t in range(seq_len - 1, -1, -1):
            x_t = x_seq[t]
            h_b, c_b = self.lstm_backward(x_t, (h_b, c_b))
            backward_states.insert(0, h_b)
            
        outputs = []
        for t in range(seq_len):
            combined_h = torch.cat((forward_states[t], backward_states[t]), dim=1)
            out = self.fc_out(combined_h)
            outputs.append(out)
            
        return torch.stack(outputs)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 3. Data Preparation
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

# ==========================================
# 4. Training Loop 
# ==========================================
def train_blstm(model, names, char_to_idx, vocab_size, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nStarting BLSTM training (Prefix Method) for {epochs} epochs...")
    
    for epoch in range(epochs):
        random.shuffle(names)
        total_loss = 0
        total_steps = 0
        
        for name in names:
            # We want to predict every character, ending with <EOS>
            target_word = list(name) + ['<EOS>']
            current_prefix = ['<SOS>']
            
            model.zero_grad()
            loss = 0
            
            # Train the model by feeding it growing prefixes
            for target_char in target_word:
                input_tensor = sequence_to_tensor(current_prefix, char_to_idx, vocab_size)
                target_idx = torch.tensor([char_to_idx[target_char]])
                
                outputs = model(input_tensor)
                
                # Take the prediction from the LAST time step of the current prefix
                prediction = outputs[-1, 0, :].unsqueeze(0)
                
                loss += criterion(prediction, target_idx)
                
                # Grow the prefix for the next step
                current_prefix.append(target_char)
                total_steps += 1
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Average Loss: {total_loss / total_steps:.4f}")
            
    return model

# ==========================================
# 5. Name Generation 
# ==========================================
def generate_name(model, char_to_idx, idx_to_char, vocab_size, max_length=15, temperature=0.8):
    with torch.no_grad():
        current_seq = ['<SOS>']
        output_name = ""
        
        for _ in range(max_length):
            seq_tensor = sequence_to_tensor(current_seq, char_to_idx, vocab_size)
            output = model(seq_tensor)
            
            last_timestep_output = output[-1, 0, :]
            
            output_dist = last_timestep_output / temperature
            probabilities = torch.softmax(output_dist, dim=0)
            topi = torch.multinomial(probabilities, 1).item()
            
            next_char = idx_to_char[topi]
            
            if next_char == '<EOS>':
                break
                
            output_name += next_char
            current_seq.append(next_char)
            
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
    
    blstm_model = BLSTMScratch(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE)
    
    print("--- BLSTM Model Details ---")
    print(f"Hyperparameters:")
    print(f"  - Hidden Size: {HIDDEN_SIZE}")
    print(f"  - Layers: {LAYERS} (Bidirectional)")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"Total Trainable Parameters: {count_parameters(blstm_model)}")
    
    trained_blstm = train_blstm(blstm_model, names, char_to_idx, VOCAB_SIZE, epochs=EPOCHS, lr=LEARNING_RATE)
    
    print("\n--- Generated BLSTM Name Samples ---")
    for _ in range(10):
        print(generate_name(trained_blstm, char_to_idx, idx_to_char, VOCAB_SIZE))