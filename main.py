import os
import pandas as pd
import pdfplumber
import re
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForLanguageModeling, 
    Trainer, TrainingArguments)
import torch
from sklearn.model_selection import train_test_split

# Extract Text from Each PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


pdf_paths = [
             "./documents/Constitutia.pdf", "./documents/CodCivil.pdf", "./documents/CodPenal.pdf",
             "./documents/CodulMuncii.pdf", "./documents/CodProceduraCivila.pdf", 
             "./documents/CodProceduraPenala.pdf", "./documents/CodFiscal.pdf",
             "./documents/LegeaSanatatii.pdf", "./documents/LegeaEducatiei.pdf",
             "./documents/ModelCIM.pdf"
             ]
texts = [extract_text_from_pdf(pdf_path) for pdf_path in pdf_paths]

# Preprocess the Text
def preprocess_text(text):
    # Remove unnecessary whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text).strip()
    return text

clean_texts = [preprocess_text(text) for text in texts]

# Combine all texts into a single string
combined_text = " ".join(clean_texts)

# Split the combined text into manageable chunks
def split_text(text, chunk_size=512):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

chunks = split_text(combined_text)

# Convert the text chunks into a pandas DataFrame
df = pd.DataFrame(chunks, columns=['text'])

# Convert to Dataset
dataset = Dataset.from_pandas(df)

# Save the dataset to a CSV file (optional)
dataset.to_csv("romanian_legal_dataset.csv", index=False)

# Split the dataset into training and validation sets
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']



# Tokenize the Combined Dataset

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Equall/Saul-Instruct-v1")

# Set pad_token to eos_token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
dataset = load_dataset('csv', data_files='romanian_legal_dataset.csv')

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Shift the logits and labels for evaluation
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss)
    return {"perplexity": perplexity.item(), "loss": loss.item()}

# Fine-tune the Model

model_name = "Equall/Saul-Instruct-v1"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate the model using the validation dataset
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the fine-tuned model
model.save_pretrained("fine-tuned-saullm-7b-romanian-legal")
tokenizer.save_pretrained("fine-tuned-saullm-7b-romanian-legal")