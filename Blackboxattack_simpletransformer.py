#using simple transformer ///// cerebras and with attack model 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
import math
import os
from tqdm import tqdm
import time

print(os.getcwd())

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Cerebras GPT-111M model and tokenizer
print("Loading model and tokenizer...")
model = GPT2LMHeadModel.from_pretrained("cerebras/Cerebras-GPT-111M")
tokenizer = GPT2Tokenizer.from_pretrained("cerebras/Cerebras-GPT-111M")

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

model.to(device)
print("Model and tokenizer loaded successfully.")

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Define the adversarial attack function
def adversarial_attack(input_prompt, target_model, num_iterations=100, step_size=0.01):
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to(device)
    input_embeds = target_model.embedding(input_ids).detach()
    input_embeds.requires_grad = True

    for _ in tqdm(range(num_iterations), desc="Adversarial attack"):
        output = target_model(inputs_embeds=input_embeds)
        target_output = torch.roll(output, shifts=-1, dims=1)
        target_output[:, -1, :] = output[:, -1, :]
        loss = F.cross_entropy(output.view(-1, output.size(-1)), target_output.view(-1, output.size(-1)).argmax(dim=-1))
        loss.backward()

        if input_embeds.grad is not None:
            input_embeds.data += step_size * input_embeds.grad.data.sign()
            input_embeds.grad.zero_()
        else:
            print("Warning: Gradient is None. Skipping update.")

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
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1.):
        self.model.eval()
        device = start_tokens.device
        num_dims = len(start_tokens.shape)
        if num_dims == 1:
            start_tokens = start_tokens[None, :]
        b, t = start_tokens.shape
        prev_out = start_tokens
        for _ in range(seq_len):
            x = prev_out[:, -self.max_seq_len:]
            logits = self.model(x).logits
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            sample = torch.multinomial(probs, 1)
            prev_out = torch.cat((prev_out, sample), dim=-1)
            if eos_token is not None and (sample == eos_token).all():
                break
        out = prev_out[:, start_tokens.shape[1]:]
        if num_dims == 1:
            out = out.squeeze(0)
        return out

    def forward(self, x=None, inputs_embeds=None, **kwargs):
        return self.model(x=x, inputs_embeds=inputs_embeds, **kwargs)

# Training loop with gradient accumulation, learning rate scheduling, and early stopping
def train_local_llm(model, data, optimizer, scheduler, num_epochs=100, batch_size=4, accumulation_steps=4):
    model.train()
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(total=len(data), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, input_ids in enumerate(data):
            input_ids = input_ids.to(device)
            output = model(input_ids)  # Remove .logits here
            output = output.view(-1, output.size(-1))
            target = input_ids.view(-1)
            
            if output.size(0) != target.size(0):
                min_size = min(output.size(0), target.size(0))
                output = output[:min_size, :]
                target = target[:min_size]
            
            loss = loss_fn(output, target)
            loss = loss / accumulation_steps  # Normalize the loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(data):
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': total_loss / (i + 1)})
        
        progress_bar.close()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data):.4f}")
        
        scheduler.step(total_loss / len(data))
        
        # Early stopping
        val_loss = total_loss / len(data)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping")
            break
        
        time.sleep(0.1)  # Add a small delay for better visual separation

# Function to read stories from file
def read_stories_from_file(file_path, delimiter="<END>"):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    stories = content.split(delimiter)
    return [story.strip() for story in stories if story.strip()]

# Function to read and preprocess WikiText-103 dataset
def read_wikitext(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Split the text into sentences or paragraphs as needed
    documents = text.split('\n\n')  # Adjust this based on the file structure
    return [doc.strip() for doc in documents if doc.strip()]

# Function to encode stories in batches
def encode_stories(stories, tokenizer, device, batch_size=4):
    encoded_stories = []
    for i in tqdm(range(0, len(stories), batch_size), desc="Encoding stories"):
        batch = stories[i:i+batch_size]
        try:
            encoded_batch = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            encoded_stories.append(encoded_batch.input_ids.to(device))
        except ValueError as e:
            print(f"Error encoding batch: {e}")
            print("Skipping this batch and continuing...")
    return encoded_stories

# Function to generate story
def generate_story(prompt, model, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, seq_len=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Function to calculate perplexity
def calculate_perplexity(model, text):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids)  # Remove .logits here
        shift_logits = outputs[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return math.exp(loss.item())

# Load and encode WikiText-103 dataset
print("\nLoading and encoding WikiText-103 dataset...")
wikitext_path = r'C:/Users/vchan/Desktop/BlackBoxAttack/BlackBoxAttack/wikitext-103/wikitext-103/wiki.train.tokens'
wikitext_data = read_wikitext(wikitext_path)
print(f"Number of documents in the dataset: {len(wikitext_data)}")
encoded_wikitext = encode_stories(wikitext_data, tokenizer, device)
print("WikiText-103 dataset encoded successfully.")

# Pre-train the model on WikiText-103
print("\nPre-training the model on WikiText-103...")
pretrain_optimizer = optim.Adam(model.parameters(), lr=0.001)
pretrain_scheduler = get_linear_schedule_with_warmup(pretrain_optimizer, num_warmup_steps=100, num_training_steps=len(encoded_wikitext) * 10)
train_local_llm(model, encoded_wikitext, pretrain_optimizer, pretrain_scheduler, num_epochs=10, batch_size=4, accumulation_steps=4)
print("Pre-training completed.")

# Read and encode children's stories
print("\nReading and encoding children's stories...")
file_path = r'C:/Users/vchan/Desktop/BlackBoxAttack/BlackBoxAttack/stories.txt'  # Replace with your file path
children_stories = read_stories_from_file(file_path)
print(f"Number of stories read: {len(children_stories)}")
encoded_children_stories = encode_stories(children_stories, tokenizer, device)
print("Children's stories encoded successfully.")

# Fine-tune the SimpleTransformer on children's stories
print("\nFine-tuning SimpleTransformer on children's stories...")
simple_transformer = SimpleTransformer(
    dim=512,  # Increased from 256
    num_unique_tokens=tokenizer.vocab_size,
    num_layers=8,  # Increased from 6
    heads=16,  # Increased from 8
    max_seq_len=128,
    causal=True
).to(device)

optimizer = optim.Adam(simple_transformer.parameters(), lr=0.001)
total_steps = len(encoded_children_stories) * 100  # Assuming 100 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

train_local_llm(simple_transformer, encoded_children_stories, optimizer, scheduler, num_epochs=100, batch_size=4, accumulation_steps=4)
print("Fine-tuning completed.")

# Set up for story generation and adversarial attacks
print("\nSetting up for story generation and adversarial attacks...")
ar_wrapper = AutoRegressiveWrapper(simple_transformer).to(device)

story_prompts = [
    "Once upon a time, in a magical forest,",
    "There was a brave little mouse named",
    "On a sunny day, the friendly dragon",
    "In a small village, a kind wizard lived",
    "The playful puppy found a mysterious"
]

adversarial_prompts = [
    "Suddenly, a scary monster appeared",
    "The children were lost in the dark",
    "Everything exploded with a loud bang",
    "A mean witch cast an evil spell",
    "The ground shook and cracked open"
]

# Generate stories and perform adversarial attacks
print("\nGenerating stories and performing adversarial attacks...")
for prompt in story_prompts:
    print(f"\n{'='*50}")
    print(f"Original prompt: {prompt}")
    generated_story = generate_story(prompt, ar_wrapper)
    print(f"Generated story: {generated_story}")
    original_perplexity = calculate_perplexity(ar_wrapper, generated_story)
    print(f"Original Perplexity: {original_perplexity:.2f}")
    
    for adv_prompt in adversarial_prompts:
        print(f"\n{'-'*40}")
        print(f"Adversarial prompt: {adv_prompt}")
        adversarial_output = adversarial_attack(prompt + " " + adv_prompt, ar_wrapper)
        print(f"Adversarial output: {adversarial_output}")
        
        adversarial_perplexity = calculate_perplexity(ar_wrapper, adversarial_output)
        print(f"Adversarial Perplexity: {adversarial_perplexity:.2f}")
    
    time.sleep(0.5)  # Add a small delay for better visual separation

print("\nProcess completed.")
