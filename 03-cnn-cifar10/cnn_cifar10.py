"""CNN sur CIFAR-10 avec PyTorch.

Entraîne un petit CNN (2 convolutions + 2 couches denses) sur CIFAR-10,
évalue le taux de reconnaissance sur le jeu de test, puis sauvegarde :
  - les courbes d'apprentissage,
  - les feature maps des deux couches de convolution pour une image donnée,
  - un fichier results.json avec les hyperparamètres et les scores.

Exemple :
    python cnn_cifar10.py --epochs 15
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Statistiques par canal (R, G, B) de CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("avion", "auto", "oiseau", "chat", "cerf",
           "chien", "grenouille", "cheval", "bateau", "camion")


class MyCNN(nn.Module):
    """conv(3→32) → ReLU → dropout → conv(32→32) → ReLU → maxpool → FC(512) → FC(10)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 512)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = self.dropout2(self.relu3(self.fc1(self.flatten(x))))
        return self.fc2(x)

    @torch.no_grad()
    def feature_maps(self, x):
        """Sorties (avant ReLU) des deux convolutions, en suivant le vrai chemin du réseau."""
        was_training = self.training
        self.eval()
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.dropout1(self.relu1(conv1_out)))
        self.train(was_training)
        return conv1_out, conv2_out


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_loaders(data_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainset, trainloader, testloader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Retourne la loss moyenne et l'accuracy (%) sur l'époque (dropout actif)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, labels) in enumerate(loader, start=1):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 200 == 0 or batch_idx == len(loader):
            print(f"  batch {batch_idx}/{len(loader)} - loss moyenne = {running_loss / total:.4f}", flush=True)
    return running_loss / total, 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Retourne la loss moyenne et l'accuracy (%) en mode eval (dropout désactivé)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        running_loss += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100 * correct / total


def plot_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["test_loss"], label="test")
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["test_acc"], label="test")
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Taux de reconnaissance")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_maps(maps, title, path):
    """Affiche les 32 cartes de caractéristiques dans une grille 4x8."""
    maps = maps[0].cpu()
    fig = plt.figure(figsize=(12, 6.5))
    for i in range(maps.shape[0]):
        plt.subplot(4, 8, i + 1)
        plt.imshow(maps[i], cmap="gray")
        plt.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_feature_map_figures(model, trainset, index, device, out_dir):
    # Image brute pour l'affichage, image transformée (normalisée) pour le réseau
    raw_image = trainset.data[index]
    tensor, label = trainset[index]
    conv1_out, conv2_out = model.feature_maps(tensor.unsqueeze(0).to(device))

    plt.figure(figsize=(3, 3))
    plt.imshow(raw_image)
    plt.title(f"Image {index} du train : {CLASSES[label]}")
    plt.axis("off")
    plt.savefig(out_dir / "input_image.png", dpi=150, bbox_inches="tight")
    plt.close()

    plot_feature_maps(conv1_out, "Feature maps de la 1re couche de convolution (32 filtres)",
                      out_dir / "feature_maps_conv1.png")
    plot_feature_maps(conv2_out, "Feature maps de la 2e couche de convolution (32 filtres)",
                      out_dir / "feature_maps_conv2.png")


def parse_args():
    parser = argparse.ArgumentParser(description="CNN sur CIFAR-10")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./results")
    parser.add_argument("--image-index", type=int, default=9,
                        help="Indice de l'image du train utilisée pour les feature maps")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device : {device}")

    print("Chargement de CIFAR-10...", flush=True)
    trainset, trainloader, testloader = get_loaders(args.data_dir, args.batch_size, args.num_workers)
    print(f"Données prêtes : {len(trainset)} images d'entraînement.", flush=True)

    model = MyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        for key, value in zip(history, (train_loss, train_acc, test_loss, test_acc)):
            history[key].append(value)
        print(f"Epoch {epoch}/{args.epochs} : "
              f"train loss = {train_loss:.4f}, train acc = {train_acc:.2f}% | "
              f"test loss = {test_loss:.4f}, test acc = {test_acc:.2f}%")

    final_acc = history["test_acc"][-1]
    print(f"Taux de reconnaissance final sur le test = {final_acc:.2f}%")

    plot_curves(history, out_dir / "training_curves.png")
    save_feature_map_figures(model, trainset, args.image_index, device, out_dir)
    torch.save(model.state_dict(), out_dir / "cnn_cifar10.pt")
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "final_test_acc": final_acc, "history": history}, f, indent=2)
    print(f"Résultats sauvegardés dans {out_dir.resolve()}")


if __name__ == "__main__":
    main()
