import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import CamembertForTokenClassification, CamembertTokenizerFast

# Load the model and tokenizer
model = CamembertForTokenClassification.from_pretrained('camembert-base', num_labels=2)
model.load_state_dict(torch.load('model.pth'))
tokenizer = CamembertTokenizerFast.from_pretrained('camembert-base')

# Load datasets
with open('train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)

with open('eval_dataset.pkl', 'rb') as f:
    eval_dataset = pickle.load(f)

# Load training logs
with open('trainer_logs.pkl', 'rb') as f:
    log_history = pickle.load(f)

# Function to plot class distribution
def plot_class_distribution(pause_count, no_pause_count, title="Class Distribution"):
    plt.figure(figsize=(6, 4))
    plt.bar(['Pause', 'No Pause'], [pause_count, no_pause_count], color=['blue', 'green'])
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()

# Plot original and balanced distribution
original_pause_count = sum(label == 1 for label_seq in train_dataset['labels'] for label in label_seq if label != -100)
original_no_pause_count = sum(label == 0 for label_seq in train_dataset['labels'] for label in label_seq if label != -100)
print(f"Original Pause count: {original_pause_count}, Original No pause count: {original_no_pause_count}")

plot_class_distribution(original_pause_count, original_no_pause_count, title="Class Distribution")

# Function to plot training metrics
def plot_training_metrics(log_history):
    # Extract loss and accuracy from training logs
    losses = [log["loss"] for log in log_history if "loss" in log]
    steps = [log["step"] for log in log_history if "loss" in log]

    # Plot loss over steps
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, label='Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss over Steps')
    plt.legend()
    plt.show()
    
    # Plot accuracy if available
    accuracies = [log["eval_accuracy"] for log in log_history if "eval_accuracy" in log]
    eval_steps = [log["step"] for log in log_history if "eval_accuracy" in log]

    if accuracies:
        plt.figure(figsize=(8, 4))
        plt.plot(eval_steps, accuracies, label='Evaluation Accuracy')
        plt.xlabel('Evaluation Steps')
        plt.ylabel('Accuracy')
        plt.title('Evaluation Accuracy over Steps')
        plt.legend()
        plt.show()

# Plot training metrics
plot_training_metrics(log_history)

# Function to plot confusion matrix
def plot_confusion_matrix(eval_dataset, model, tokenizer):
    all_predictions = []
    all_labels = []
    
    for batch in eval_dataset:
        inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in batch.items() if k != 'labels'}
        labels = torch.tensor(batch['labels']).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Flatten predictions and labels
        predictions = predictions.view(-1)
        labels = labels.view(-1)
        
        # Filter out -100 labels
        mask = labels != -100
        filtered_predictions = predictions[mask]
        filtered_labels = labels[mask]
        
        all_predictions.extend(filtered_predictions.tolist())
        all_labels.extend(filtered_labels.tolist())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    labels = ['No Pause', 'Pause']
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Plot confusion matrix
plot_confusion_matrix(eval_dataset, model, tokenizer)
