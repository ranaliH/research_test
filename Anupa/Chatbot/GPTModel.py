import torch
import torch.nn as nn
from torch.nn import functional as F
import json


class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[nn.TransformerEncoderLayer(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        return logits


def train_model(train_data, vocab_size, n_embd, block_size, n_head, n_layer, max_iters, batch_size, eval_interval, learning_rate):
    # Set up the model
    model = GPTModel(vocab_size, n_embd, block_size, n_head, n_layer)
    model = model.to(device)

    # Create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for iter in range(max_iters):
        # Sample a batch of data
        batch_indices = torch.randint(0, len(train_data), (batch_size,))
        batch_patterns = [train_data[idx][0] for idx in batch_indices]
        batch_responses = [train_data[idx][1] for idx in batch_indices]

        # Find the maximum length within the batch
        max_len = max(max(len(pattern), len(response)) for pattern, response in zip(batch_patterns, batch_responses))

        # Pad patterns and responses to the maximum length
        padded_patterns = [pattern + [0] * (max_len - len(pattern)) for pattern in batch_patterns]
        padded_responses = [response + [0] * (max_len - len(response)) for response in batch_responses]

        # Convert to tensors
        xb = torch.tensor(padded_patterns, dtype=torch.long, device=device)
        yb = torch.tensor(padded_responses, dtype=torch.long, device=device)

        # Adjust the position_embedding_table size
        model.position_embedding_table = nn.Embedding(max_len, n_embd).to(device)

        # Evaluate the loss
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss
        if iter % eval_interval == 0 or iter == max_iters - 1:
            print(f"Step {iter}: loss {loss.item():.4f}")

    return model


def generate_response(model, input_pattern, block_size):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        input_pattern_encoded = encode(input_pattern)
        input_pattern_tensor = torch.tensor(input_pattern_encoded, dtype=torch.long, device=device).unsqueeze(0)
        B, T = input_pattern_tensor.shape
        generated_response = input_pattern_encoded  # Initialize the generated response with the input pattern

        for _ in range(block_size):
            logits = model(input_pattern_tensor)
            logits = logits[:, -1, :]  # Get the logits for the last token in the sequence
            predicted_token = torch.argmax(logits, dim=-1)  # Select the token with the highest probability

            # Append the predicted token to the generated response
            generated_response.append(predicted_token.item())

            # Update the input pattern tensor for the next iteration
            input_pattern_tensor = torch.cat([input_pattern_tensor, predicted_token.unsqueeze(1)], dim=1)

        generated_response = decode(generated_response)
        return generated_response


if __name__ == "__main__":
    batch_size = 16
    block_size = 512
    max_iters = 400  # Updated value
    eval_interval = 100
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 64
    n_head = 4
    n_layer = 4

    torch.manual_seed(1337)
    # Read intents data
    with open('intents_modified.json', 'r', encoding='utf-8') as f:
        intents_data = json.load(f)

    # Create a mapping from characters to integers
    patterns = []
    responses = []
    tag2idx = {}
    idx2tag = {}
    for idx, intent in enumerate(intents_data['intents']):
        tag = intent['tag']
        tag2idx[tag] = idx
        idx2tag[idx] = tag
        patterns.extend(intent['patterns'])
        responses.extend(intent['responses'])

    chars = sorted(list(set(''.join(patterns + responses))))  # Include characters from both patterns and responses
    vocab_size = len(chars)

    # Add '<unk>' token to the characters dictionary
    chars.append('<unk>')
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encoder and decoder functions with '<unk>' handling
    encode = lambda s: [stoi.get(c, stoi['<unk>']) for c in s]  # Encoder: takes a string, outputs a list of integers, uses '<unk>' for unknown characters
    decode = lambda l: ''.join([itos[i] for i in l])  # Decoder: takes a list of integers, outputs a string

    # Encode the patterns and responses
    encoded_patterns = [encode(pattern) for pattern in patterns]
    encoded_responses = [encode(response) for response in responses]

    # Create train dataApologies for the confusion. Here's the updated code:


import torch
import torch.nn as nn
from torch.nn import functional as F
import json


class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[nn.TransformerEncoderLayer(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        return logits


def train_model(train_data, vocab_size, n_embd, block_size, n_head, n_layer, max_iters, batch_size, eval_interval, learning_rate):
    # Set up the model
    model = GPTModel(vocab_size, n_embd, block_size, n_head, n_layer)
    model = model.to(device)

    # Create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for iter in range(max_iters):
        # Sample a batch of data
        batch_indices = torch.randint(0, len(train_data), (batch_size,))
        batch_patterns = [train_data[idx][0] for idx in batch_indices]
        batch_responses = [train_data[idx][1] for idx in batch_indices]

        # Find the maximum length within the batch
        max_len = max(max(len(pattern), len(response)) for pattern, response in zip(batch_patterns, batch_responses))

        # Pad patterns and responses to the maximum length
        padded_patterns = [pattern + [0] * (max_len - len(pattern)) for pattern in batch_patterns]
        padded_responses = [response + [0] * (max_len - len(response)) for response in batch_responses]

        # Convert to tensors
        xb = torch.tensor(padded_patterns, dtype=torch.long, device=device)
        yb = torch.tensor(padded_responses, dtype=torch.long, device=device)

        # Adjust the position_embedding_table size
        model.position_embedding_table = nn.Embedding(max_len, n_embd).to(device)

        # Evaluate the loss
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss
        if iter % eval_interval == 0 or iter == max_iters - 1:
            print(f"Step {iter}: loss {loss.item():.4f}")

    return model


def generate_response(model, input_pattern, block_size):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        input_pattern_encoded = encode(input_pattern)
        input_pattern_tensor = torch.tensor(input_pattern_encoded, dtype=torch.long, device=device).unsqueeze(0)
        B, T = input_pattern_tensor.shape
        generated_response = input_pattern_encoded  # Initialize the generated response with the input pattern

        for _ in range(block_size):
            logits = model(input_pattern_tensor)
            logits = logits[:, -1, :]  # Get the logits for the last token in the sequence
            predicted_token = torch.argmax(logits, dim=-1)  # Select the token with the highest probability

            # Append the predicted token to the generated response
            generated_response.append(predicted_token.item())

            # Update the input pattern tensor for the next iteration
            input_pattern_tensor = torch.cat([input_pattern_tensor, predicted_token.unsqueeze(1)], dim=1)

        generated_response = decode(generated_response)
        return generated_response


if __name__ == "__main__":
    batch_size = 16
    block_size = 512
    max_iters = 400  # Updated value
    eval_interval = 100
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 64
    n_head = 4
    n_layer = 4

    torch.manual_seed(1337)
    # Read intents data
    with open('intents_modified.json', 'r', encoding='utf-8') as f:
        intents_data = json.load(f)

    # Create a mapping from characters to integers
    patterns = []
    responses = []
    tag2idx = {}
    idx2tag = {}
    for idx, intent in enumerate(intents_data['intents']):
        tag = intent['tag']
        tag2idx[tag] = idx
        idx2tag[idx] = tag
        patterns.extend(intent['patterns'])
        responses.extend(intent['responses'])

    chars = sorted(list(set(''.join(patterns + responses))))  # Include characters from both patterns and responses
    vocab_size = len(chars)
    # Add '<unk>' token to the characters dictionary
    chars.append('<unk>')
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encoder and decoder functions with '<unk>' handling
    encode = lambda s: [stoi.get(c, stoi['<unk>']) for c in s]  # Encoder: takes a string, outputs a list of integers, uses '<unk>' for unknown characters
    decode = lambda l: ''.join([itos[i] for i in l])  # Decoder: takes a list of integers, outputs a string

    # Encode the patterns and responses
    encoded_patterns = [encode(pattern) for pattern in patterns]
    encoded_responses = [encode(response) for response in responses]

    # Create train data
    train_data = []
    for pattern, response in zip(encoded_patterns, encoded_responses):
        train_data.append((pattern, response))

    # Train the model
    model = train_model(train_data, vocab_size, n_embd, block_size, n_head, n_layer, max_iters, batch_size, eval_interval, learning_rate)

    torch.save(model.state_dict(), 'gpt_model.pth')

    # input_pattern = "Hi there"
    # generated_response = generate_response(model, input_pattern, block_size)
    # print(f"Input: {input_pattern}")
    # print(f"Generated Response: {generated_response}")
