import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5Model
from transformers import PegasusTokenizer
from torch.utils.data import DataLoader, Dataset

# Load your dataset (replace with your data loading logic)
data = json.loads(open('intents_modified.json').read())

# Prepare text for language modeling (combine patterns and responses)
text_corpus = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        text_corpus.append(pattern)
    for response in intent["responses"]:
        text_corpus.append(response)

# Load tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5Model.from_pretrained('t5-small')

# Tokenize and encode the text
inputs = tokenizer(text_corpus, return_tensors="pt", padding=True, truncation=True)

# Create a custom dataset
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

dataset = TextDataset(inputs)

# Create data loader
batch_size = 8
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone().detach()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss.item():.4f}")
