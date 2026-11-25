"""MLP sur le jeu de données Wine : cross-entropy vs MSE, avec ou sans BatchNorm.

Pour chaque essai (split train/test différent, stratifié), le script entraîne
quatre réseaux identiques à l'initialisation près :
  - cross-entropy sans BatchNorm      - cross-entropy avec BatchNorm
  - MSE sans BatchNorm                - MSE avec BatchNorm
puis affiche la moyenne et l'écart-type du taux de reconnaissance sur le test.

Le StandardScaler est ajusté sur le jeu d'entraînement uniquement, pour ne pas
laisser d'information du test fuiter dans le prétraitement.

Exemple :
    python mlp_wine.py --trials 10 --epochs 1000
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
    """Deux couches cachées de même taille. Retourne les logits (pas de softmax)."""

    def __init__(self, input_size, hidden_size, output_size, batchnorm=False):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, hidden_size))
        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(model, x_train, y_train, loss_name, epochs, lr, num_classes):
    """Entraînement en batch complet.

    - "ce"  : CrossEntropyLoss sur les logits (le softmax est déjà inclus dans la loss).
    - "mse" : MSE entre les probabilités (softmax des logits) et la cible one-hot.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    y_one_hot = F.one_hot(y_train, num_classes).float()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        if loss_name == "ce":
            loss = criterion_ce(logits, y_train)
        else:
            loss = criterion_mse(F.softmax(logits, dim=1), y_one_hot)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    predictions = model(x).argmax(dim=1)
    return (predictions == y).float().mean().item()


def parse_args():
    parser = argparse.ArgumentParser(description="MLP sur Wine : CE vs MSE, avec/sans BatchNorm")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="./results")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_wine()
    X, y = data.data, data.target
    num_classes = len(data.target_names)

    configs = [(loss, bn) for bn in (False, True) for loss in ("ce", "mse")]
    names = {(loss, bn): f"{loss.upper()}{' + BatchNorm' if bn else ''}" for loss, bn in configs}
    scores = {names[c]: [] for c in configs}

    for trial in range(args.trials):
        seed = args.seed + trial
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, shuffle=True, stratify=y, random_state=seed
        )

        # Le scaler ne voit que le train
        scaler = StandardScaler().fit(X_train)
        x_train = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
        x_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        line = []
        for loss_name, bn in configs:
            set_seed(seed)  # même initialisation de départ pour chaque config
            model = MLP(X.shape[1], args.hidden, num_classes, batchnorm=bn)
            train(model, x_train, y_train_t, loss_name, args.epochs, args.lr, num_classes)
            acc = accuracy(model, x_test, y_test_t)
            scores[names[(loss_name, bn)]].append(acc)
            line.append(f"{names[(loss_name, bn)]} = {acc:.4f}")
        print(f"Essai {trial + 1}/{args.trials} : " + " | ".join(line), flush=True)

    print("\nRésultats (moyenne ± écart-type sur le test) :")
    summary = {}
    for name, values in scores.items():
        mean, std = float(np.mean(values)), float(np.std(values))
        summary[name] = {"mean": mean, "std": std, "scores": values}
        print(f"  {name:<18} : {mean:.4f} ± {std:.4f}")

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=2)
    print(f"Résultats sauvegardés dans {out_dir.resolve()}")


if __name__ == "__main__":
    main()
