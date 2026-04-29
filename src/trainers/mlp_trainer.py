import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()

        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        assert x.ndim == 2
        assert x.shape[1] == self.input_dim
        return self.net(x)


def make_loaders(X, y, batch_size=32, val_fraction=0.2, seed=123):
    dataset = TensorDataset(X, y)

    n = len(dataset)
    n_val = int(n * val_fraction)
    n_train = n - n_val

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    total_n = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.shape[0]
        total_n += xb.shape[0]

    return total_loss / total_n


def evaluate(model, val_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            total_loss += loss.item() * xb.shape[0]
            total_n += xb.shape[0]

    return total_loss / total_n


def train_model(
    model,
    X,
    y,
    max_epochs=100,
    batch_size=32,
    lr=1e-3,
    patience=10,
    log_path="artifacts/day10/train_log.csv",
    ckpt_path="artifacts/day10/best_model.pt",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    train_loader, val_loader = make_loaders(
        X=X,
        y=y,
        batch_size=batch_size,
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    bad_epochs = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                train_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )

            val_loss = evaluate(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )

            writer.writerow([epoch, train_loss, val_loss])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                bad_epochs += 1

            if bad_epochs >= patience:
                break

    return best_val_loss


if __name__ == "__main__":
    torch.manual_seed(123)

    X = torch.randn(300, 10)
    true_w = torch.randn(10, 1)
    y = X @ true_w + 0.1 * torch.randn(300, 1)

    model = MLP(
        input_dim=10,
        hidden_dim=64,
        output_dim=1,
        dropout=0.2,
    )

    best_val_loss = train_model(
        model=model,
        X=X,
        y=y,
        max_epochs=100,
        batch_size=32,
        lr=1e-3,
        patience=10,
    )

    print("best_val_loss:", best_val_loss)