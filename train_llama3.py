import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback
from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
from transformers.models.llama import LlamaForCausalLM
from datasets import Dataset
from huggingface_hub import login

# Set your Hugging Face API token
token = "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"

# Login to Hugging Face
login(token=token)

# Dataset
dataset_string = """The Magical Seed
In a small village, there was a boy named Sam who loved plants. One day, he found a mysterious seed in the forest. He planted it in his garden and watered it every day. To his surprise, the seed grew into a giant beanstalk overnight. Sam climbed the beanstalk and discovered a land above the clouds. There, he met a friendly giant who showed him a garden full of magical plants.
The Lost Kitten
Emily was walking home from school when she heard a faint meowing. She followed the sound and found a tiny kitten hiding in a bush. The kitten looked lost and scared. Emily took the kitten home, fed it, and gave it a warm bed. The next day, she put up posters around the neighborhood. A few days later, the kitten's owner saw the poster and came to collect their lost pet. Emily felt happy to reunite the kitten with its family.
The Brave Little Turtle
Tommy the turtle was afraid of water. All his friends swam in the pond, but Tommy stayed on the shore. One day, he saw a duckling struggling in the water. Without thinking, Tommy jumped in and pushed the duckling to safety. His friends cheered for him, and Tommy realized that he was braver than he thought.
The Rainbow Fish
In a colorful coral reef, there lived a fish named Rainbow. Rainbow had shiny scales that glittered in all the colors of the rainbow. But Rainbow was lonely because the other fish were jealous of his beauty. One day, a small fish asked Rainbow for one of his scales. Rainbow hesitated but then gave the small fish a glittering scale. The small fish was overjoyed and told all the other fish about Rainbow's kindness. Soon, Rainbow had many friends, and he was no longer lonely.
The Curious Caterpillar
Carl the caterpillar loved exploring the garden. One day, he saw a butterfly fluttering in the sky. Curious, Carl asked the butterfly how it had grown wings. The butterfly explained that it was once a caterpillar, just like Carl. Excited, Carl ate lots of leaves and spun a cocoon around himself. After some time, he emerged as a beautiful butterfly and flew into the sky."""

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = self.load_dataset(dataset)

    def load_dataset(self, dataset):
        try:
            data = {'text': dataset}
            dataset = Dataset.from_dict(data)
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def __len__(self):
        return len(self.dataset['text'])

    def __getitem__(self, index):
        text = self.dataset['text'][index]
        return text

def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=tokenizer.model_max_length)

def main(rank, world_size, tokenizer):
    try:
        # Set up model and tokenizer
        model_name = "meta-llama/Llama-2-7b-chat-hf"

        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Initialize distributed training
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # Split the dataset into individual stories
        stories = [story.strip() for story in dataset_string.split('\n\n')]

        # Load and preprocess the dataset
            # Process the dataset in smaller chunks
        chunk_size = 2  # Adjust this value as needed
        for i in range(0, len(stories), chunk_size):
            chunk = stories[i:i+chunk_size]

            # Load and preprocess the dataset chunk
            dataset = MyDataset(chunk)

            # Check if dataset is loaded
            if dataset is None or dataset.dataset is None:
                raise ValueError("Dataset is not loaded properly")

            tokenized_dataset = dataset.dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

            # Check if tokenized dataset is None
            if tokenized_dataset is None:
                raise ValueError("Tokenized dataset is not created properly")

            # Set up training arguments
            training_args = TrainingArguments(
                output_dir='./llama3-sft',
                per_device_train_batch_size=1,
                learning_rate=0.0001,
                max_steps=1000,
                logging_steps=10,
                gradient_checkpointing=True,
                fp16=True,
                report_to=[],
            )

            # Set up data collator
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            # Set up model
            model = LlamaForCausalLM.from_pretrained(model_name)
            model = model.to(rank)

            # Set up trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )

            # Start training
            trainer.train()

    except Exception as e:
        print(f"Rank {rank} encountered an error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Set up model and tokenizer
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Reduce the sequence length
    tokenizer.model_max_length = 128

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if CUDA is available
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    mp.spawn(main, args=(world_size, tokenizer), nprocs=world_size, join=True)