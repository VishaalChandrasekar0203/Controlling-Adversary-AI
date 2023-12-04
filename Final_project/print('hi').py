import re
import torch
import numpy as np
from itertools import islice
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_n_gram_sequences(token_list):
    for i in range(1, len(token_list)):
        yield token_list[:i+1]

# Read the entire content of the file
# Specify the encoding when opening the file
with open(r"C:\Users\vchan\Desktop\adv AI\fairy_tales.txt", "r", encoding="utf-8") as df:
    stories = df.read()

stories = re.sub(r'[^\w\s]', '', stories.lower())

# Tokenize the stories
tokenizer = Tokenizer()
tokenizer.fit_on_texts([stories])  # Pass a list of strings to fit_on_texts

total_words = len(tokenizer.word_index) + 1

# Create input sequences using a generator
input_sequences_generator = generate_n_gram_sequences(tokenizer.texts_to_sequences([stories])[0])

# Process sequences in small batches
batch_size = 1000
for batch_sequences in iter(lambda: list(islice(input_sequences_generator, batch_size)), []):
    # Pad sequences
    max_sequence_length = max(len(seq) for seq in batch_sequences)
    input_sequences_padded = pad_sequences(batch_sequences, maxlen=max_sequence_length, padding='pre')

    # Convert to numpy array for further processing
    input_sequences_array = np.array(input_sequences_padded)

    # Check if input_sequences_array is not empty before proceeding
    if input_sequences_array.size > 0:
        X, y = input_sequences_array[:, :-1], input_sequences_array[:, -1]

        # Load pre-trained GPT-2 model and tokenizer
        model_name = 'gpt2'
        tokenizer_gpt2 = GPT2Tokenizer.from_pretrained(model_name)
        model_gpt2 = GPT2LMHeadModel.from_pretrained(model_name)

        # Convert input sequences to tensors
        input_ids = torch.tensor(X)
        labels = torch.tensor(y)

        # Reshape labels to match the expected shape
        labels = labels.view(-1)

        # Define PyTorch Dataset and DataLoader
        dataset = torch.utils.data.TensorDataset(input_ids, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        # Rest of your code remains unchanged...


        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_gpt2.parameters(), lr=0.001)

        # Training loop
        for epoch in range(epochs):
            for batch in dataloader:
                inputs, labels = batch
                optimizer.zero_grad()

        # Ensure labels have the correct shape
                labels = labels.view(-1)

                outputs = model_gpt2(inputs, labels=labels)
                Loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels)
                loss.backward()
                optimizer.step()

        # Save the trained model
        model_gpt2.save_pretrained('bedtime_story_model')

        def generate_bedtime_story(prompt, model, tokenizer, max_length=100):
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(generated_text)

    else:
        # Handle the case when input_sequences_array is empty
        print("Input sequences list is empty. Please check your data.")
