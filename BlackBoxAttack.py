import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math
import os

print(os.getcwd())

# Load the Cerebras GPT-111M model and tokenizer
model = GPT2LMHeadModel.from_pretrained("cerebras/Cerebras-GPT-111M")
tokenizer = GPT2Tokenizer.from_pretrained("cerebras/Cerebras-GPT-111M")

# Set the device (CPU)
device = torch.device("cpu")
model.to(device)

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Define the adversarial attack function
def adversarial_attack(input_prompt, target_model, num_iterations=100, step_size=0.01):
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to(device)
    input_embeds = target_model.model.embedding(input_ids).detach()
    input_embeds.requires_grad = True

    for _ in range(num_iterations):
        output = target_model(inputs_embeds=input_embeds)
        
        # Compute loss with a shifted target
        target_output = torch.roll(output, shifts=-1, dims=1)
        target_output[:, -1, :] = output[:, -1, :]
        loss = F.cross_entropy(output.view(-1, output.size(-1)), target_output.view(-1, output.size(-1)).argmax(dim=-1))
        
        # Backward pass
        loss.backward()
        
        # Update input_embeds
        if input_embeds.grad is not None:
            input_embeds.data += step_size * input_embeds.grad.data.sign()
            input_embeds.grad.zero_()
        else:
            print("Warning: Gradient is None. Skipping update.")

    # Convert perturbed embeddings back to token ids
    with torch.no_grad():
        logits = target_model(inputs_embeds=input_embeds)
        adversarial_ids = torch.argmax(logits, dim=-1)
    
    adversarial_output = tokenizer.decode(adversarial_ids[0], skip_special_tokens=True)
    return adversarial_output

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embedding_dim)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Multi-Head Self-Attention
class MHSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, causal=True):
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.causal = causal
        self.to_qkv = nn.Linear(dim, _dim * 3, bias=False)
        self.W_out = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        b, seq_len, _ = x.shape
        qkv = self.to_qkv(x)
        q, k, v = tuple(qkv.chunk(3, dim=-1))
        q, k, v = map(lambda tensor: tensor.view(b, seq_len, self.heads, self.dim_head).transpose(1, 2), (q, k, v))
        scaled_dot_prod = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale_factor
        i, j = scaled_dot_prod.shape[2:]
        if self.causal:
            causal_mask = torch.triu(torch.ones(i, j, device=x.device), diagonal=1).bool()
            scaled_dot_prod = scaled_dot_prod.masked_fill(causal_mask, float('-inf'))
        if mask is not None:
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask[:, None, None, :], float('-inf'))
        attention = torch.softmax(scaled_dot_prod, dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attention, v)
        out = out.transpose(1, 2).contiguous().view(b, seq_len, -1)
        return self.W_out(out)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, causal=False, dim_linear_block=1024, dropout=0.1):
        super().__init__()
        self.mhsa = MHSelfAttention(dim=dim, heads=heads, dim_head=dim_head, causal=causal)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        y = self.norm_1(self.mhsa(x, mask) + x)
        return self.norm_2(self.linear(y) + y)

# Transformer
class Transformer(nn.Module):
    def __init__(self, dim, num_layers, heads, max_seq_len, causal=True):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, heads, causal=causal) for _ in range(num_layers)])
        self.pos_emb = PositionalEncoding(dim, max_seq_len)

    def forward(self, x):
        x = self.pos_emb(x)
        for layer in self.layers:
            x = layer(x)
        return x

# Simple Transformer
class SimpleTransformer(nn.Module):
    def __init__(self, dim, num_unique_tokens, num_layers, heads, max_seq_len, causal=True):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(num_unique_tokens, dim)
        self.transformer = Transformer(dim, num_layers, heads, max_seq_len, causal=causal)
        self.fc = nn.Linear(dim, num_unique_tokens)
        self.max_seq_len = max_seq_len

    def forward(self, x=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(x)
        x = self.transformer(inputs_embeds)
        x = self.fc(x)
        return x

# AutoRegressive Wrapper
class AutoRegressiveWrapper(nn.Module):
    def __init__(self, net, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.model = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1., filter_thres=0.9):
        self.model.eval()
        device = start_tokens.device
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape
        prev_out = start_tokens

        for _ in range(seq_len):
            x = prev_out[:, -self.max_seq_len:]
            logits = self.model(x)
            logits = logits[:, -1, :] / temperature
            filtered_logits = top_k(logits, thres=filter_thres)
            sample = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
            prev_out = torch.cat((prev_out, sample), dim=-1)

            if eos_token is not None and (sample == eos_token).all():
                break

        out = prev_out[:, start_tokens.shape[1]:]

        if num_dims == 1:
            out = out.squeeze(0)

        return out

    def forward(self, x=None, inputs_embeds=None, **kwargs):
        return self.model(x=x, inputs_embeds=inputs_embeds, **kwargs)

# Training loop
def train_local_llm(model, data, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for input_ids in data:
            input_ids = input_ids.to(device)
            optimizer.zero_grad()
            output = model(input_ids)
            
            # Reshape output and target
            output = output.view(-1, output.size(-1))
            target = input_ids.view(-1)
            
            # Ensure output and target have the same first dimension
            if output.size(0) != target.size(0):
                min_size = min(output.size(0), target.size(0))
                output = output[:min_size, :]
                target = target[:min_size]
            
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data)}")

# Prepare data
shakespeare_data = [
    "To be or not to be, that is the question.",
    "All the world's a stage, and all the men and women merely players."
]
encoded_data = [torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device) for text in shakespeare_data]

# Model parameters
vocab_size = tokenizer.vocab_size
embedding_dim = 256
num_layers = 6
heads = 8
max_seq_len = 128

# Create and train SimpleTransformer
simple_transformer = SimpleTransformer(
    dim=embedding_dim,
    num_unique_tokens=vocab_size,
    num_layers=num_layers,
    heads=heads,
    max_seq_len=max_seq_len,
    causal=True
).to(device)

optimizer = optim.Adam(simple_transformer.parameters(), lr=0.001)
train_local_llm(simple_transformer, encoded_data, optimizer, num_epochs=100)

# AutoRegressiveWrapper for generation
ar_wrapper = AutoRegressiveWrapper(simple_transformer).to(device)

# Adversarial attack example
input_prompt = "Shall I compare thee to a summer's day?"
adversarial_output = adversarial_attack(input_prompt, ar_wrapper)
print("Adversarial Output:", adversarial_output)
