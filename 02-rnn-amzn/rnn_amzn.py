"""LSTM vs GRU : prédiction du cours de clôture d'Amazon (AMZN, 2006-2017).

Pour chaque couple (lookback, taille cachée), le script entraîne un LSTM et un GRU
à deux couches, puis compare leur erreur sur le jeu de test à une baseline
« le prix de demain = le prix d'aujourd'hui ».

Choix de méthode (voir le README) :
  - Découpage chronologique : 80 % du temps pour l'entraînement, 20 % pour le test.
  - Chaque fenêtre de prix est exprimée en variations relatives par rapport à son
    dernier prix : x / dernier_prix - 1. Le réseau n'a donc pas à extrapoler à des
    niveaux de prix jamais vus à l'entraînement (le prix est multiplié par ~25 entre
    2006 et 2017). Aucun scaler n'est ajusté sur le test : il n'y a pas de fuite.
  - Les erreurs sont mesurées en dollars, après retour au prix.

Exemple de test rapide :
    python rnn_amzn.py --lookbacks 10 --hidden-dims 32 --n-seeds 1 --epochs 5

Exemple complet :
    python rnn_amzn.py
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------- données

def make_windows(prices, lookback, train_frac):
    """Construit les fenêtres glissantes et le découpage chronologique.

    Entrée d'un exemple : `lookback` prix consécutifs.  Cible : le prix suivant.
    Les entrées et la cible sont exprimées relativement au dernier prix de la fenêtre.
    Une fenêtre est dans le test si son prix cible tombe dans les derniers 20 % de la série.
    """
    n = len(prices)
    split = int(n * train_frac)
    idx = np.arange(n - lookback)
    windows = np.stack([prices[i:i + lookback] for i in idx])
    last = windows[:, -1]
    target_price = prices[idx + lookback]

    x_rel = windows / last[:, None] - 1.0
    y_rel = target_price / last - 1.0

    is_train = (idx + lookback) < split
    scale = y_rel[is_train].std()  # calculé sur le train uniquement

    return {
        "x_train": x_rel[is_train] / scale,
        "y_train": y_rel[is_train] / scale,
        "x_test": x_rel[~is_train] / scale,
        "last_test": last[~is_train],
        "target_test": target_price[~is_train],
        "scale": scale,
    }


# ---------------------------------------------------------------- modèle

class RecurrentModel(nn.Module):
    """LSTM ou GRU à `num_layers` couches, suivi d'une couche linéaire.

    Les états initiaux sont mis à zéro automatiquement par PyTorch.
    """

    def __init__(self, kind, hidden_dim, num_layers=2):
        super().__init__()
        rnn_class = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_class(1, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model, x_train, y_train, epochs, lr, batch_size):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = x_train.size(0)
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def test_mse_dollars(model, data):
    """MSE en dollars² : on repasse de la variation relative au prix prédit."""
    model.eval()
    x_test = torch.tensor(data["x_test"], dtype=torch.float32).unsqueeze(-1)
    pred_rel = model(x_test).squeeze(1).numpy() * data["scale"]
    pred_price = data["last_test"] * (1.0 + pred_rel)
    return float(np.mean((pred_price - data["target_test"]) ** 2))


def baseline_mse_dollars(data):
    """Baseline naïve : le prix prédit est le dernier prix connu."""
    return float(np.mean((data["last_test"] - data["target_test"]) ** 2))


# ---------------------------------------------------------------- programme

def parse_args():
    parser = argparse.ArgumentParser(description="LSTM vs GRU sur le cours d'AMZN")
    parser.add_argument("--csv", default="./AMZN_2006-01-01_to_2018-01-01.csv")
    parser.add_argument("--lookbacks", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Nombre d'entraînements par configuration (résultats moyennés)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="./results")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"Fichier introuvable : {csv_path.resolve()}\n"
                 "Place AMZN_2006-01-01_to_2018-01-01.csv dans ce dossier "
                 "(ou indique son chemin avec --csv).")

    prices = pd.read_csv(csv_path)["Close"].to_numpy(dtype=np.float64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(prices)} jours de cotation, entraînement sur les {args.train_frac:.0%} premiers.", flush=True)

    rows = []
    for lookback in args.lookbacks:
        data = make_windows(prices, lookback, args.train_frac)
        x_train = torch.tensor(data["x_train"], dtype=torch.float32).unsqueeze(-1)
        y_train = torch.tensor(data["y_train"], dtype=torch.float32).unsqueeze(-1)
        baseline = baseline_mse_dollars(data)

        for hidden_dim in args.hidden_dims:
            scores = {"lstm": [], "gru": []}
            for kind in scores:
                for s in range(args.n_seeds):
                    set_seed(args.seed + s)
                    model = RecurrentModel(kind, hidden_dim)
                    train_model(model, x_train, y_train, args.epochs, args.lr, args.batch_size)
                    scores[kind].append(test_mse_dollars(model, data))

            row = {
                "lookback": lookback,
                "hidden_dim": hidden_dim,
                "lstm_mse": float(np.mean(scores["lstm"])),
                "gru_mse": float(np.mean(scores["gru"])),
                "baseline_mse": baseline,
                "lstm_mse_all_seeds": scores["lstm"],
                "gru_mse_all_seeds": scores["gru"],
            }
            rows.append(row)
            print(f"Lookback {lookback:<3} | Hidden {hidden_dim:<3} | "
                  f"LSTM MSE {row['lstm_mse']:8.2f} | GRU MSE {row['gru_mse']:8.2f} | "
                  f"Baseline MSE {baseline:8.2f}", flush=True)

    best_lstm = min(rows, key=lambda r: r["lstm_mse"])
    best_gru = min(rows, key=lambda r: r["gru_mse"])
    print("\nRésumé (MSE en dollars² sur le test, moyenne sur les graines) :")
    print(f"  Baseline (prix de la veille) : {rows[0]['baseline_mse']:.2f} "
          f"(RMSE ≈ {rows[0]['baseline_mse'] ** 0.5:.2f} $) pour lookback {rows[0]['lookback']}")
    print(f"  Meilleur LSTM : {best_lstm['lstm_mse']:.2f} (lookback {best_lstm['lookback']}, "
          f"hidden {best_lstm['hidden_dim']})")
    print(f"  Meilleur GRU  : {best_gru['gru_mse']:.2f} (lookback {best_gru['lookback']}, "
          f"hidden {best_gru['hidden_dim']})")

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    print(f"Résultats sauvegardés dans {out_dir.resolve()}")


if __name__ == "__main__":
    main()
