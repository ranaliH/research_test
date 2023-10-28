from GPTModel import *
import torch
import torch.nn as nn

batch_size = 16
block_size = 512
max_iters = 400
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 64
n_head = 4
n_layer = 4
vocab_size = 81

# Load the checkpoint
checkpoint = torch.load('gpt_model.pth', map_location=device)

# Extract the vocabulary size from the checkpoint
vocab_size = checkpoint['token_embedding_table.weight'].shape[0]

# Create the model with the correct vocabulary size
model = GPTModel(vocab_size, n_embd, block_size, n_head, n_layer)

# Update the position_embedding_table weight shape in the model to match the checkpoint
position_emb_shape = checkpoint['position_embedding_table.weight'].shape
model.position_embedding_table = nn.Embedding(position_emb_shape[0], position_emb_shape[1])

# Load the state dict from the checkpoint
model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

# Generate response
input_pattern = "Hi there"
generated_response = generate_response(model, input_pattern, block_size)
print(f"Input: {input_pattern}")
print(f"Generated Response: {generated_response}")
