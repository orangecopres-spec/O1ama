import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1. Simple Tokenizer
# -----------------------------
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] if c in self.stoi else self.stoi[' '] for c in s]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)


# -----------------------------
# 2. Dataset
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size+1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# -----------------------------
# 3. Transformer Model (causal)
# -----------------------------
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4*n_embd,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.block_size = block_size

    def causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        tok = self.token_emb(idx)
        pos = self.pos_emb(pos)
        x = tok + pos

        mask = self.causal_mask(T).to(idx.device)
        x = self.transformer(x, mask=mask)
        logits = self.lm_head(x)
        return logits


# -----------------------------
# 4. Training / Eval
# -----------------------------
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)


# -----------------------------
# 5. Greedy Text Generation
# -----------------------------
def generate(model, tokenizer, device, start_text="", max_new_tokens=40):
    model.eval()
    with torch.no_grad():
        if start_text == "":
            start_ids = [0]
        else:
            start_ids = tokenizer.encode(start_text)

        x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            if x.size(1) > model.block_size:
                x_cond = x[:, -model.block_size:]
            else:
                x_cond = x

            logits = model(x_cond)
            logits = logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            x = torch.cat([x, next_id], dim=1)

        out_ids = x[0].tolist()
        return tokenizer.decode(out_ids)


# -----------------------------
# 6. Main
# -----------------------------
if __name__ == "__main__":
    text = """
user: what is 2 + 2
assistant: 4
user: what is 3 + 3
assistant: 6
user: what is 4 + 4
assistant: 8
user: what is 5 + 5
assistant: 10
user: what is 6 + 6
assistant: 12
user: what is 7 + 7
assistant: 14
user: what is 8 + 8
assistant: 16
user: what is 9 + 9
assistant: 18
user: what is 10 + 10
assistant: 20
user: what is 11 + 11
assistant: 22
user: what is 12 + 12
assistant: 24
user: what is 13 + 13
assistant: 26
user: what is 14 + 14
assistant: 28
user: what is 15 + 15
assistant: 30
user: what is 16 + 16
assistant: 32
user: what is 17 + 17
assistant: 34
user: what is 18 + 18
assistant: 36
user: what is 19 + 19
assistant: 38
user: what is 20 + 20
assistant: 40
user: hello
assistant: hello
user: are you alive
assistant: yes
user: who created you
assistant: debiancoder https://github.com/orangecopres-spec
"""

    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)

    block_size = 64
    batch_size = 16
    lr = 3e-4
    epochs = 30

    dataset = TextDataset(data, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerModel(tokenizer.vocab_size, block_size=block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = train(model, loader, optimizer, device)
        val_loss = evaluate(model, loader, device)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    torch.save(model.state_dict(), "transformer.pth")
    print("Model saved to transformer.pth")

    print("\n=== Chat Mode ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        user_input = user_input.lower().strip()

        # Force exact training pattern
        prompt = f"user: what is {user_input}\nassistant: "

        reply_full = generate(model, tokenizer, device, start_text=prompt, max_new_tokens=10)
        reply = reply_full[len(prompt):].strip().split("\n")[0]

        # Fallback if model doesn't know the answer
        if len(reply) == 0 or not any(ch.isdigit() for ch in reply):
            reply = "hello io ewr"

        print("O1ama:", reply)
