import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset

# Load your CSV file into a DataFrame
file_path = 'C:/Users/VishalChandrasekar/OneDrive - WSD Digital, LLC/Desktop/Generative AI/CF_PP_2025.csv'  # Replace with the path to your dataset
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1)

# Check dataset size
dataset_size = len(df)
print(f"Dataset size: {dataset_size}")

# Check for missing values in the dataset
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print("Missing values in the dataset:")
    print(missing_values)

# Convert float values to strings in selected columns
float_columns = ['RECIPNAME', 'NAME', 'CITY', 'STATE', 'OCCUPATION', 'EMPNAME', 'EMPCITY']
df[float_columns] = df[float_columns].astype(str)

# Concatenate multiple columns into a single text column
df['text'] = df[float_columns].agg(' '.join, axis=1)

# Initialize GPT-2 tokenizer and model
model_name = 'gpt2-medium'  # Change to 'gpt2' for the base model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Manually set the padding token
tokenizer.pad_token = tokenizer.eos_token  # You can choose another token as the pad token if needed

# Tokenize your dataset
tokenized_data = tokenizer.batch_encode_plus(
    df['text'].tolist(),  # Use the 'text' column that contains concatenated text
    add_special_tokens=True,
    max_length=280,
    padding=True,
    truncation=True,
    return_tensors='pt',
)

# Create a dataset from the tokenized data
dataset = Dataset.from_dict(tokenized_data)

# Initialize a data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./fine_tuned_gpt2_model',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize a Trainer instance for training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,  # Use the 'dataset' variable here
)

# Train the model
try:
    trainer.train()
except Exception as e:
    print("An error occurred during training:")
    print(e)

# Save the trained model
try:
    trainer.save_model()
except Exception as e:
    print("An error occurred while saving the model:")
    print(e)
