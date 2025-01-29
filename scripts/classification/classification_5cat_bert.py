import random
import torch
import pandas as pd
from transformers import CamembertTokenizerFast, CamembertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset

# Function to extract labels and clean text
def get_pause_category(text):
    global_labels = []
    clean_text = []
    sentences = text.split(".")
    for sentence in sentences:
        words = sentence.split(" ")
        labels = []
        for i in range(len(words)):
            if words[i] in ["cat_1", "cat_2", "cat_3", "cat_4", "cat_5"]:
                labels.append(int(words[i].split("_")[1]))
            else:
                labels.append(0)
                clean_text.append(words[i])
        global_labels.append(labels)
    return global_labels, clean_text  # Return clean_text as a list of words

def align_labels_with_tokens(tokenizer, words, labels):
    # Tokenize the words with padding and truncation to ensure equal length
    tokens = tokenizer(words, return_tensors="pt", is_split_into_words=True, padding="max_length", max_length=128, truncation=True)
    word_ids = tokens.word_ids()  # Word indices that map back to words in the original sentence
    aligned_labels = []

    print(f"Original words: {words}")
    print(f"Original labels: {labels}")
    print(f"Word IDs: {word_ids}")

    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:  # Special tokens like [CLS] or [SEP]
            aligned_labels.append(-100)  # Ignored during loss calculation
        elif word_id != previous_word_id:  # Start of a new word
            if word_id < len(labels):  # Check if word_id is within bounds
                aligned_labels.append(labels[word_id])  # Assign the correct label
            else:
                aligned_labels.append(-100)  # Handle out-of-bounds index
        else:
            if word_id < len(labels):  # Check if word_id is within bounds
                aligned_labels.append(labels[word_id])  # Assign the same label to subwords
            else:
                aligned_labels.append(-100)  # Handle out-of-bounds index
        previous_word_id = word_id

    # Ensure the labels are padded to match the sequence length
    while len(aligned_labels) < 128:
        aligned_labels.append(-100)  # Padding for labels too

    print(f"Aligned labels: {aligned_labels}")
    return tokens, aligned_labels


def print_predictions(tokens, predictions, tokenizer):
    # Convert token ids back to words
    decoded_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0], skip_special_tokens=True)

    # Map label indices back to the pause categories
    label_map = {0: 'O', 1: 'cat_1', 2: 'cat_2', 3: 'cat_3', 4: 'cat_4', 5: 'cat_5'}
    predicted_labels = [label_map.get(p, 'O') for p in predictions]

    # Print tokens and their corresponding predicted labels
    for token, label in zip(decoded_tokens, predicted_labels):
        print(f"Token: {token}, Predicted Label: {label}")

# Set a random seed
random_seed = 42
random.seed(random_seed)

# Set a random seed for PyTorch (for GPU as well)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# Use the fast tokenizer
tokenizer = CamembertTokenizerFast.from_pretrained('camembert-base')
model = CamembertForTokenClassification.from_pretrained('camembert-base')

# Example text with pause categories
text = "cat_2 L'intention de l cat_5 'aéroport de biard  cat_5 de  cat_4 diminuer  cat_1 la poussé des gaz sur le décollage de ses avions  cat_1 au dessus  cat_5 des zones  cat_5 habité  cat_3 à ses avantage et ses inconvéniant."

# Get labels and clean text
labels, clean_words = get_pause_category(text)

# Tokenize and align labels
tokens, aligned_labels = align_labels_with_tokens(tokenizer, clean_words, labels[0])

# Convert aligned labels to tensor format
input_ids = tokens['input_ids'].tolist()
attention_mask = tokens['attention_mask'].tolist()

# Step 1: Create a Pandas DataFrame, ensure same length for all columns
df = pd.DataFrame({
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'labels': [aligned_labels]  # Wrap aligned_labels in a list for a single row
})

print("DataFrame:\n", df.head())

# Step 2: Convert the DataFrame to a Dataset
dataset = Dataset.from_pandas(df)

print("Hugging Face Dataset:\n", dataset)

# Step 3: Define training arguments and Trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_dir="./logs",
    no_cuda=True
)

# Step 4: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Step 5: Start training
trainer.train()

model.eval()
with torch.no_grad():
    # Perform a forward pass
    outputs = model(**tokens)
predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
print_predictions(tokens, predictions, tokenizer)