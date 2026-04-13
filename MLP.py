import torch.nn as nn
import torch
import sklearn

class MLP(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=3, dropout=0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.LazyLinear(hidden_dim),          # input_dim inferred at first forward
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)
    
def train_mlp(
    train_loader,
    val_loader,
    hidden_dim=256,
    num_classes=3,
    dropout=0.3,
    num_epochs=100,
    lr=1e-3,
    save_path="mlp.pt"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model     = MLP(hidden_dim, num_classes, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):

        # ── Training ──────────────────────────────
        model.train()
        train_loss    = 0.0
        train_correct = 0
        n_train       = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            n_train       += len(y_batch)

        torch.save(model.state_dict(), save_path)

    print(f"\nBest val loss: {best_val_loss:.5f}")
    return model

def evaluate_mlp(model, loader):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    return all_labels, all_preds