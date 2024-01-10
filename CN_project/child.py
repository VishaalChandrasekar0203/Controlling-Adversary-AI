import numpy as np
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Read the entire content of the file
with open("/Users/vishaalchandrasekar/Desktop/CN_project/fairy_tales.txt", "r") as df:
    stories = df.read()

stories = re.sub(r'[^\w\s]', '', stories.lower())

# Tokenize the stories
tokenizer = Tokenizer()
tokenizer.fit_on_texts([stories])  # Pass a list of strings to fit_on_texts

total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
token_list = tokenizer.texts_to_sequences([stories])[0]
for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Check if input_sequences is not empty before proceeding
if input_sequences.size > 0:
    X, y = input_sequences[:, :-1], input_sequences[:, -1]

    # Load pre-trained GPT-2 model and tokenizer
    model_name = 'gpt2'
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained(model_name)
    model_gpt2 = GPT2LMHeadModel.from_pretrained(model_name)

    # Convert input sequences to tensors
    input_ids = torch.tensor(X)
    labels = torch.tensor(y)

    # Define PyTorch Dataset and DataLoader
    dataset = torch.utils.data.TensorDataset(input_ids, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_gpt2.parameters(), lr=0.001)

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        for batch in dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model_gpt2(inputs, labels=labels)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    # Save the trained model
    model_gpt2.save_pretrained('bedtime_story_model')

    def generate_bedtime_story(prompt, model, tokenizer, max_length=100):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    prompt = "Once upon a time in a magical land,"
    generated_story = generate_bedtime_story(prompt, model_gpt2, tokenizer_gpt2)
    print(generated_story)

else:
    # Handle the case when input_sequences is empty
    print("Input sequences list is empty. Please check your data.")



