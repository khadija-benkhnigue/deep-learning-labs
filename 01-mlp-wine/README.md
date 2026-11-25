# TP1 : MLP sur le jeu de données Wine (PyTorch)

Classification de 178 vins en 3 classes à partir de 13 mesures chimiques, avec un perceptron multicouche (MLP). Le TP compare deux fonctions de coût (cross-entropy et MSE) et l'effet de la normalisation par lots (BatchNorm).

## Objectif et données

- **Objectif** : prendre en main PyTorch sur un problème de classification multiclasse avec un perceptron multicouche. Il s'agit de prédire, à partir de 13 mesures chimiques, de quel type de vin il s'agit parmi 3 classes.
- **Données** : jeu de données Wine (UCI), fourni avec scikit-learn (`load_wine`) : 178 exemples, 13 variables numériques, 3 classes (trois cultivars d'une même région d'Italie).
- **Évaluation** : taux de reconnaissance (accuracy) sur un jeu de test de 40 % des données, moyenné sur 10 découpages train/test différents, avec son écart-type.

## Résultats

Moyenne ± écart-type du taux de reconnaissance sur le jeu de test, sur 10 essais :

| Variante | Taux de reconnaissance |
|---|---|
| Cross-entropy | 97,36 % ± 1,15 |
| MSE | 97,08 % ± 0,97 |
| Cross-entropy + BatchNorm | 97,08 % ± 0,75 |
| MSE + BatchNorm | 96,67 % ± 0,92 |

### Interprétation

Les quatre variantes atteignent environ 97 %. Les écarts entre elles (0,7 point au maximum) sont plus petits que les écarts-types et que la résolution du jeu de test : avec 72 vins de test, **une seule erreur représente 1,4 point**. On ne peut donc pas conclure qu'une variante est meilleure qu'une autre sur ce jeu de données : Wine est un problème facile, et les quatre réseaux le résolvent aussi bien.

## Protocole

- 10 essais, chacun avec un découpage différent (graines 42 à 51) : 60 % train / 40 % test, stratifié par classe.
- Normalisation `StandardScaler` **ajustée sur le train uniquement**, puis appliquée au test, pour éviter toute fuite d'information.
- Pour un essai donné, les quatre variantes utilisent le même découpage et la même initialisation de départ : la comparaison est équitable.
- Réseau : 13 entrées → 50 → 50 → 3 sorties, activations ReLU. Avec BatchNorm, une couche de normalisation est ajoutée après chaque couche cachée.
- Entraînement en batch complet, Adam, learning rate 1e-4, 1000 époques.

### Les deux fonctions de coût

- **Cross-entropy** : appliquée directement sur les sorties brutes du réseau (logits). Le softmax est inclus dans la loss de PyTorch.
- **MSE** : appliquée entre les probabilités (softmax des sorties) et la classe cible codée en one-hot.

## Lancer le code

```bash
pip install -r ../requirements.txt
python mlp_wine.py
```

Options : `--trials`, `--epochs`, `--hidden`, `--lr`, `--test-size`, `--seed`, `--out-dir`. Le jeu de données est fourni avec scikit-learn, aucun téléchargement n'est nécessaire. Les scores de chaque essai sont enregistrés dans `results/results.json`.

## Limites

- Jeu de test très petit (72 exemples) : les scores sont peu précis.
- Pas de réglage des hyperparamètres (taille cachée, learning rate) : ceux de l'énoncé du TP ont été conservés.
- Aucun test statistique entre les variantes, vu l'absence d'écart net.
