import numpy as np

from atomic.activations import *
from atomic.layers.conv import Conv2D, Flatten, MaxPool2D
from atomic.models.custom import BaseModel
from atomic.layers.dense import Dense, parse_dtype
from atomic.losses import *
from atomic.optimizers import *
from atomic.models.sequential import Sequential
from atomic.core.autograd.tensor import Tensor, has_cupy
try:
    import cupy as cp # type: ignore
except ImportError:
    cp = None
#-==


class Embedding(BaseModel):
    def __init__(self, vocab_size, d_model, dtype=np.float32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dtype = dtype
        # Xavier initialization using parent's device/dtype
        init_std = np.sqrt(2.0 / (vocab_size + d_model))
        weight_data = np.random.randn(vocab_size, d_model).astype(self.dtype) * init_std
        
        self.weight = Tensor(weight_data, 
                           device=self.device,  # Use BaseModel's device
                           dtype=self.dtype,
                           requires_grad=True)

    def forward(self, x):
        # Ensure input is Tensor and on correct device
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device, dtype=np.int64)
        else:
            x = x.to(self.device).astype(np.int64)  # Force device/dtype
        return self.weight[x]
    


class PositionalEncoding(BaseModel):
    def __init__(self, d_model: int, max_seq_len: int = 5000, dtype=np.float32):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pos_enc = self._create_positional_encoding()
        self.dtype = dtype

    def _create_positional_encoding(self):
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(-(np.arange(0, self.d_model, 2) * np.log(10000.0) / self.d_model))
        
        pos_enc = np.zeros((self.max_seq_len, self.d_model), dtype=self.dtype)
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        
        return Tensor(pos_enc, 
                     device=self.device,
                     requires_grad=False,
                     dtype=self.dtype)

    def forward(self, x: Tensor) -> Tensor:
        # Get proper numerical library
        xp = x.xp
        
        # Get positional encoding for sequence length
        seq_len = x.shape[1]
        pos_enc = self.pos_enc[:seq_len]
        
        # Reshape positional encoding for broadcasting
        # From (seq_len, d_model) to (1, seq_len, d_model)
        pos_enc = pos_enc.reshape(1, seq_len, self.d_model)
        
        # Expand to match batch dimension
        # Result shape: (batch_size, seq_len, d_model)
        pos_enc = xp.broadcast_to(pos_enc.data, (x.shape[0], seq_len, self.d_model))
        
        return x + Tensor(pos_enc, device=self.device, dtype=self.dtype)

class MultiHeadAttention(BaseModel):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, dtype=np.float32):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.dtype = dtype
        # All layers inherit device/dtype from parent
        self.Wq = Dense(d_model, d_model,dtype=self.dtype)
        self.Wk = Dense(d_model, d_model,dtype=self.dtype)
        self.Wv = Dense(d_model, d_model,dtype=self.dtype)
        self.Wo = Dense(d_model, d_model,dtype=self.dtype)

    def __call__(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        return self.forward(query, key, value, mask)

    def split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)\
               .transpose(0, 2, 1, 3)

    def combine_heads(self, x: Tensor) -> Tensor:
        return x.transpose(0, 2, 1, 3)\
               .reshape(x.shape[0], -1, self.d_model)

    def scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.where(mask, -1e9)
        attn_weights = Softmax(axis=-1)(attn_scores)
        return attn_weights @ v

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        q = self.split_heads(self.Wq(query))
        k = self.split_heads(self.Wk(key))
        v = self.split_heads(self.Wv(value))
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        return self.Wo(self.combine_heads(attn_output))



class LayerNorm(BaseModel):
    def __init__(self, features: int, eps: float = 1e-5, dtype=np.float32):
        """
        Layer Normalization
        
        Args:
            features: Size of the input feature dimension
            eps: Small value to prevent division by zero
            dtype: Data type for parameters
        """
        super().__init__()
        self.features = features
        self.eps = eps
        self.dtype = dtype

        # Learnable parameters
        self.gamma = Tensor(np.ones(features), 
                           dtype=dtype,
                           requires_grad=True)
        self.beta = Tensor(np.zeros(features),
                          dtype=dtype,
                          requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # Ensure input is on correct device
        x = x.to(self.device).astype(self.dtype)
        
        # Calculate statistics
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True, unbiased=False)  # Use biased variance
        
        # Normalize
        x_normalized = (x - mean) / (var + self.eps).sqrt()
        #print('====================>')
        #print(self.gamma.dtype)
        #print(self.gamma.device)
        #print(x_normalized.dtype)
        #print(x_normalized.device)
        # Scale and shift
        return self.gamma * x_normalized + self.beta

    @property
    def parameters(self):
        """Return gamma and beta for optimization"""
        return [self.gamma, self.beta]

    def state_dict(self):
        return {
            "gamma": self.gamma.data.copy(),
            "beta": self.beta.data.copy(),
            "features": self.features,
            "eps": self.eps,
            "device": self.device
        }

    def load_state_dict(self, state_dict):
        self.gamma.data = state_dict['gamma']
        self.beta.data = state_dict['beta']
        self.features = state_dict.get('features', self.features)
        self.eps = state_dict.get('eps', self.eps)


class Encoder(BaseModel):
    def __init__(self, vocab_size, d_model, num_heads, dtype=np.float32):
        super().__init__()
        self.dtype = dtype
        self.embeddings = Embedding(vocab_size, d_model, dtype=self.dtype)
        self.positional = PositionalEncoding(d_model, dtype=self.dtype)
        self.attention = MultiHeadAttention(d_model, num_heads, dtype=self.dtype)
        self.norm1 = LayerNorm(d_model, dtype=self.dtype)
        self.norm2 = LayerNorm(d_model, dtype=self.dtype)
        self.feedforward = Sequential([
            Dense(d_model, d_model * 4, dtype=self.dtype),
            ReLU(),
            Dense(d_model * 4, d_model, dtype=self.dtype)])

    def forward(self, x: Tensor) -> Tensor:
        # Embeddings and positional encoding
        x = self.embeddings(x)
        x = self.positional(x)
        
        # Attention block
        residual = x
        x = self.attention(x, x, x)
        x = residual + x
        x = self.norm1(x)
        
        # Feedforward block
        residual = x
        x = self.feedforward(x)
        x = residual + x
        x = self.norm2(x)
        
        return x
    


class Decoder(BaseModel):
    def __init__(self, vocab_size, d_model, num_heads, dtype=np.float32):
        super().__init__()
        self.dtype = dtype
        self.embeddings = Embedding(vocab_size, d_model, dtype=self.dtype)
        self.positional = PositionalEncoding(d_model, dtype=self.dtype)
        
        # Self attention with masking
        self.self_attention = MultiHeadAttention(d_model, num_heads, dtype=self.dtype)
        self.norm1 = LayerNorm(d_model, dtype=self.dtype)
        
        # Encoder-Decoder attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dtype=self.dtype)
        self.norm2 = LayerNorm(d_model, dtype=self.dtype)
        
        # Feedforward network
        self.feedforward = Sequential([
            Dense(d_model, d_model * 4, dtype=self.dtype),
            ReLU(),
            Dense(d_model * 4, d_model, dtype=self.dtype)
        ])
        self.norm3 = LayerNorm(d_model, dtype=self.dtype)

    def create_causal_mask(self, seq_len):
        """Create mask to prevent looking at future positions"""
        xp = np if self.device == 'cpu' else cp
        mask = xp.ones((1, seq_len, seq_len), dtype=self.dtype)
        mask = xp.tril(mask)  # Lower triangular matrix
        return Tensor(mask, device=self.device, requires_grad=False)

    def __call__(self, target: Tensor, encoder_output: Tensor) -> Tensor:
        return self.forward(target, encoder_output)

    def forward(self, target: Tensor, encoder_output: Tensor) -> Tensor:
        # Embed target sequence
        x = self.embeddings(target)
        x = self.positional(x)
        
        # Self attention with causal masking
        residual = x
        seq_len = x.shape[1]
        mask = self.create_causal_mask(seq_len)
        x = self.self_attention(x, x, x, mask)
        x = residual + x
        x = self.norm1(x)
        
        # Encoder-decoder attention
        residual = x
        x = self.cross_attention(x, encoder_output, encoder_output)
        x = residual + x
        x = self.norm2(x)
        
        # Feedforward block
        residual = x
        x = self.feedforward(x)
        x = residual + x
        x = self.norm3(x)
        
        return x




class Transformer(BaseModel):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff=None, num_layers=1, dropout=0.1, share_embeddings=False, dtype=np.float32):
        """
        A full Transformer model that combines an Encoder, a Decoder, and a final projection layer.
        
        Args:
            src_vocab_size (int): Vocabulary size for the encoder (source language).
            tgt_vocab_size (int): Vocabulary size for the decoder (target language).
            d_model (int): Dimensionality of the model.
            num_heads (int): Number of attention heads.
            d_ff (int, optional): Dimension of feed forward network. Defaults to 4*d_model.
            num_layers (int, optional): Number of encoder/decoder layers. Defaults to 1.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            share_embeddings (bool, optional): Whether to share embeddings. Defaults to False.
            dtype: Data type for computations.
        """
        super().__init__()
        self.dtype = dtype
        
        if d_ff is None:
            d_ff = 4 * d_model

        # Build the encoder and decoder blocks (each includes embeddings and positional encoding)
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, dtype=dtype)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, dtype=dtype)
        
        # Final linear layer to map decoder output to target vocabulary logits
        self.final_linear = Dense(d_model, tgt_vocab_size, dtype=dtype)
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        Forward pass of the Transformer.
        
        Args:
            src (Tensor): Input tensor of shape (batch_size, src_seq_len) containing token indices.
            tgt (Tensor): Target tensor of shape (batch_size, tgt_seq_len) containing token indices.
            
        Returns:
            Tensor: Logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Pass source tokens through the encoder to get a continuous representation.
        encoder_output = self.encoder(src)
        
        # Pass target tokens and the encoder output through the decoder.
        decoder_output = self.decoder(tgt, encoder_output)
        
        # Project the decoder output into the target vocabulary space.
        logits = self.final_linear(decoder_output)
        return logits

    def __call__(self, src: Tensor, tgt: Tensor) -> Tensor:
        return self.forward(src, tgt)

    def train(self):
        """Set model to training mode"""
        for module in self._modules.values():
            if hasattr(module, 'train'):
                module.train()

# Training code follows

import os
import requests
import numpy as np
import random
import pickle
import time

def run_training():
    # ---------------------
    # Step 1: Download / Load Dataset
    # ---------------------
    file_path = 'shakespeare.txt'
    if os.path.exists(file_path):
        print("Loading existing Shakespeare dataset...")
        with open(file_path, 'r', encoding='utf-8') as f:
            shakespeare_text = f.read()
    else:
        print("Downloading Shakespeare dataset...")
        url = "https://www.gutenberg.org/files/100/100-0.txt"
        response = requests.get(url)
        shakespeare_text = response.text
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(shakespeare_text)
        print("Dataset downloaded and saved as shakespeare.txt")

    # ---------------------
    # Step 2: Preprocess the Text
    # ---------------------
    shakespeare_text = shakespeare_text.lower()
    start_idx = shakespeare_text.find("the sonnets")
    end_idx = shakespeare_text.find("end of this project gutenberg")
    if start_idx != -1 and end_idx != -1:
        shakespeare_text = shakespeare_text[start_idx:end_idx].replace('\n', ' ').replace('\r', '')
    
    # Limit the text length to reduce memory usage
    max_text_length = 500000  # Use only the first 500K characters
    shakespeare_text = shakespeare_text[:max_text_length]

    # Create vocabulary and mappings
    vocab = sorted(set(shakespeare_text))
    vocab_size = len(vocab)
    char2idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}

    # Convert text to indices
    text_indices = [char2idx[ch] for ch in shakespeare_text]

    # Create input-target sequences using a stride to reduce the number of samples
    seq_len = 100
    stride = 5
    inputs = []
    targets = []
    for i in range(0, len(text_indices) - seq_len, stride):
        inputs.append(text_indices[i:i + seq_len])
        targets.append(text_indices[i + 1:i + seq_len + 1])
    inputs = np.array(inputs)
    targets = np.array(targets)

    print(f"Total characters used: {len(shakespeare_text)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of training samples: {len(inputs)}")
    print(f"Sample input: {shakespeare_text[:100]}")
    print(f"Sample target: {shakespeare_text[1:101]}")

    # ---------------------
    # Step 3: GPU Setup
    # ---------------------
    try:
        import cupy as cp  # For GPU support
        device = 'cuda'
        print("CuPy detected: Using GPU for training.")
    except ImportError:
        cp = np  # Fallback to NumPy
        device = 'cpu'
        print("CuPy not found: Using CPU for training.")

    # ---------------------
    # Step 4: Model Configuration
    # ---------------------
    # Training hyperparameters
    d_model = 128            # Model dimension
    num_heads = 4            # Number of attention heads
    num_layers = 4           # Number of encoder/decoder layers
    d_ff = d_model * 4       # Feed-forward dimension
    dropout = 0.1            # Dropout rate
    batch_size = 16          # Batch size
    num_epochs = 300          # Number of training epochs
    learning_rate = 0.001    # Initial learning rate
    warmup_epochs = 5        # Learning rate warmup epochs
    min_lr = 1e-3            # Minimum learning rate
    clip_norm = 1.0          # Gradient clipping norm
    save_path = 'shakespeare_transformer.pkl'  # Model save path

    # Instantiate model and move to proper device
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
        share_embeddings=True
    )
    model.set_device(device)
    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = Adam(model.parameters, lr=learning_rate)

    # ---------------------
    # Step 5: Define Text Generation Function
    # ---------------------
    def generate_text(model, prompt, gen_length=200, temperature=0.8):
        """
        Generate text using the model without requiring eval mode.
        """
        with Tensor.no_grad():
            # Convert prompt to character indices
            input_seq = [char2idx.get(ch, 0) for ch in prompt.lower()]
            
            # Initialize generation
            generated = list(input_seq)
            
            for _ in range(gen_length):
                # Prepare current input sequence
                curr_seq = generated[-seq_len:] if len(generated) >= seq_len else generated
                
                # Pad sequence if needed
                if len(curr_seq) < seq_len:
                    curr_seq = [0] * (seq_len - len(curr_seq)) + curr_seq
                
                # Create input tensors
                src = np.array([curr_seq])
                if device == 'cuda':
                    src = cp.array(src)
                src_tensor = Tensor(src, device=device, dtype=np.int64, requires_grad=False)
                
                # Forward pass through the model
                output = model(src_tensor, src_tensor)
                
                # Get probabilities for next character
                logits = output.data[0, -1]
                
                # Apply temperature sampling
                if temperature > 0:
                    # Convert logits to probabilities with temperature
                    probs = np.exp(logits / temperature)
                    probs = probs / np.sum(probs)
                    
                    # Sample from the distribution
                    if device == 'cuda':
                        probs = cp.asnumpy(probs)
                    next_char_idx = np.random.choice(len(probs), p=probs)
                else:
                    # Deterministic (greedy) sampling
                    if device == 'cuda':
                        next_char_idx = int(cp.argmax(logits))
                    else:
                        next_char_idx = int(np.argmax(logits))
                
                # Add the predicted character index to our sequence
                generated.append(next_char_idx)
        
        # Convert indices back to characters
        text = ''.join([idx2char[idx] for idx in generated])
        return text

    # ---------------------
    # Step 6: Training Loop with LR Scheduling, Gradient Clipping, and Sample Generation
    # ---------------------
    def shuffle_data(inputs, targets):
        """Shuffle the data while keeping inputs and targets aligned"""
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        return inputs[indices], targets[indices]

    # Track best loss for model saving
    best_loss = float('inf')
    patience = 0
    max_patience = 5  # Early stopping after 5 epochs of no improvement

    print("Starting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        # Shuffle the data for each epoch
        inputs_shuffled, targets_shuffled = shuffle_data(inputs, targets)
        
        # Adjust learning rate: warmup then exponential decay
        if epoch < warmup_epochs:
            current_lr = learning_rate * (epoch + 1) / warmup_epochs
        else:
            current_lr = max(learning_rate * (0.95 ** (epoch - warmup_epochs)), min_lr)
        optimizer.lr = current_lr  # Update optimizer's learning rate
        
        # Training metrics
        epoch_loss = 0.0
        batch_count = 0
        
        model.train()  # Ensure model is in training mode
        
        for i in range(0, len(inputs_shuffled), batch_size):
            batch_count += 1
            
            # Get mini-batch
            src_batch = inputs_shuffled[i: i + batch_size]
            tgt_batch = targets_shuffled[i: i + batch_size]
            
            # Move mini-batch data to GPU if available
            if device == 'cuda':
                src_batch = cp.array(src_batch)
                tgt_batch = cp.array(tgt_batch)
            
            # Create tensors
            src_tensor = Tensor(src_batch, device=device, dtype=np.int64, requires_grad=False)
            tgt_tensor = Tensor(tgt_batch, device=device, dtype=np.int64, requires_grad=False)
            
            # For teacher forcing: shift right for targets, left for inputs
            tgt_input = src_tensor  # Use the source as decoder input
            tgt_output = tgt_tensor  # Target is the next character
            
            # Forward pass
            logits = model(src_tensor, tgt_input)
            loss = loss_fn(logits, tgt_output)
            batch_loss = float(loss.data)
            epoch_loss += batch_loss
            
            # Print progress
            if batch_count % 20 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {batch_loss:.4f}, LR: {current_lr:.6f}")
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Gradient clipping
            for param in model.parameters:
                if param.grad is not None:
                    xp = cp if device == 'cuda' else np
                    grad_norm = xp.linalg.norm(param.grad.data)
                    if grad_norm > clip_norm:
                        param.grad.data *= clip_norm / (grad_norm + 1e-6)
            
            # Update parameters
            optimizer.step()
        
        # Calculate average loss
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save the model if it's the best

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            # Save model
            try:
                with open(save_path, 'wb') as f:
                    pickle.dump(model.state_dict(), f)
                print(f"Model saved to {save_path} (loss: {avg_loss:.4f})")
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping after {epoch+1} epochs (no improvement for {max_patience} epochs)")
                break
        
        # Generate a sample text after each epoch to monitor progress
        try:
            # Use different temperatures to see how the model behaves
            temps = [0.5, 0.7, 1.0]
            print("\n----- SAMPLES -----")
            for temp in temps:
                prompt = "o romeo, wherefore art thou"
                generated_sample = generate_text(model, prompt, gen_length=150, temperature=temp)
                print(f"\nSample (temperature={temp:.1f}):")
                print(generated_sample)
            print("----- END SAMPLES -----\n")
        except Exception as e:
            print(f"Error generating samples: {e}")

    # Training completed
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    run_training()
