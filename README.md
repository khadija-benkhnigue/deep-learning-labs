# Deep Learning Labs

Trois travaux pratiques de deep learning en PyTorch, un par grande famille de réseaux : perceptron multicouche (MLP), réseaux récurrents (LSTM/GRU) et réseau convolutif (CNN). Chaque TP est reproductible (graines fixées, options en ligne de commande) et accompagné d'un README qui détaille le protocole, les résultats et les limites.

## Les TP

| TP | Sujet | Données | Résultat principal |
|---|---|---|---|
| [01 : MLP](01-mlp-wine/) | Cross-entropy vs MSE, avec ou sans BatchNorm | Wine (178 vins, 3 classes) | ≈ 97 % de reconnaissance pour les 4 variantes ; écarts dans le bruit (10 essais) |
| [02 : RNN](02-rnn-amzn/) | LSTM vs GRU, effet du lookback et de la taille cachée | Cours d'Amazon, 2006-2017 | Aucun modèle ne bat nettement la baseline « prix de la veille » (RMSE ≈ 12,3 $) |
| [03 : CNN](03-cnn-cifar10/) | CNN à 2 convolutions, visualisation des feature maps | CIFAR-10 (60 000 images) | 69,3 % sur le test ; surapprentissage visible dès l'époque 4 |

## Démarche

- **Comparer à une référence** : le TP2 est évalué contre une baseline naïve, ce qui montre que les réseaux n'apportent pas d'amélioration réelle.
- **Éviter les fuites de données** : normalisation ajustée sur l'entraînement uniquement, découpage chronologique pour la série temporelle.
- **Ne pas sur-interpréter** : sur Wine, les écarts entre variantes sont plus petits que la résolution du jeu de test ; le README le dit.
- **Rester honnête sur les scores** : on rapporte le score de la dernière époque, pas le meilleur choisi sur le test.

## Installation

```bash
pip install -r requirements.txt
```

Chaque dossier contient son script, son README et un dossier `results/` avec les sorties (scores, courbes). Les données (CIFAR-10, fichier CSV d'Amazon) ne sont pas versionnées : voir le README de chaque TP.

```
deep-learning-labs/
├── 01-mlp-wine/        # mlp_wine.py
├── 02-rnn-amzn/        # rnn_amzn.py
├── 03-cnn-cifar10/     # cnn_cifar10.py
└── requirements.txt
```

## Environnement

Python 3.13, PyTorch, entraînements réalisés sur CPU.
