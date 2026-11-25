# TP2 : LSTM vs GRU sur le cours d'Amazon (PyTorch)

Prédiction du cours de clôture d'Amazon (AMZN) du jour suivant à partir des `lookback` jours précédents, avec un LSTM et un GRU à deux couches. Le TP explore l'effet de la longueur de la fenêtre (`lookback`) et de la taille de la couche cachée, et compare les modèles à une **baseline naïve** : « le prix de demain sera celui d'aujourd'hui ».

## Résultat principal

**Aucun des modèles ne fait nettement mieux que la baseline.**

| | MSE sur le test (dollars²) | RMSE (dollars) |
|---|---|---|
| Baseline (prix de la veille) | 152,17 | 12,34 |
| Meilleur LSTM (lookback 5, hidden 64) | 151,52 | 12,31 |
| Meilleur GRU (lookback 5, hidden 64) | 151,23 | 12,30 |

Sur les 60 résultats (30 configurations × 2 modèles), les MSE vont de 151,23 à 154,45, soit moins de 2 % d'écart autour de la baseline. Seuls 17 résultats sur 60 passent sous la baseline, et le meilleur ne la dépasse que de 0,6 %.

### Moyenne par longueur de fenêtre (sur les 5 tailles cachées)

| Lookback | LSTM | GRU |
|---|---|---|
| 5 | 151,63 | 151,65 |
| 10 | 152,30 | 152,34 |
| 15 | 152,48 | 152,59 |
| 20 | 152,61 | 152,77 |
| 25 | 153,11 | 153,07 |
| 30 | 153,21 | 153,18 |

Baseline : 152,17 (identique pour tous les lookbacks, puisque le jeu de test est le même).

### Interprétation

- **LSTM et GRU sont indiscernables** : leurs moyennes sont presque identiques (152,56 et 152,60) et chacun gagne dans la moitié des configurations.
- **Les modèles plus gros ou avec plus d'historique sont légèrement moins bons** (l'erreur moyenne passe de 151,6 à 153,2 quand le lookback passe de 5 à 30). Une explication plausible, non testée ici : les réseaux apprennent du bruit.
- Ce résultat est cohérent avec la difficulté connue de la prédiction des cours de bourse à un jour : la meilleure prédiction simple est souvent le dernier prix. Les réseaux apprennent essentiellement à recopier le dernier prix.
- Sans la baseline, une erreur de 12 $ pourrait passer pour un bon résultat : c'est la comparaison à la baseline qui montre que les réseaux n'apportent pas d'information utile ici.

## Protocole

- **Données** : 3 019 jours de cotation (2006-01-03 à 2017-12-29), colonne `Close`. Le fichier n'est pas inclus dans le dépôt.
- **Découpage chronologique** : les 80 % premiers jours pour l'entraînement (jusqu'à mi-2015 environ), les 20 % suivants pour le test (604 jours). Aucun mélange entre passé et futur.
- **Représentation des fenêtres** : chaque fenêtre de prix est exprimée en variation relative par rapport à son dernier prix (`prix / dernier_prix - 1`), la cible aussi. Le prix est multiplié par environ 25 entre 2006 et 2017 ; avec des prix bruts normalisés sur l'entraînement, le test sortirait très largement de la plage vue à l'entraînement. Cette représentation évite ce problème et ne nécessite aucun scaler ajusté sur le test.
- **Normalisation** : les valeurs sont divisées par l'écart-type des cibles de l'entraînement (calculé sur l'entraînement uniquement).
- **Erreur** : on revient au prix prédit puis on calcule la MSE en dollars² sur le jeu de test.
- **Modèles** : LSTM ou GRU à 2 couches, couche linéaire finale. Adam, learning rate 1e-3, 30 époques, mini-batchs de 128.
- **Grille** : lookback ∈ {5, 10, 15, 20, 25, 30} × taille cachée ∈ {8, 16, 32, 64, 128}.
- **Reproductibilité** : chaque configuration est entraînée avec 3 graines (42, 43, 44) et les résultats sont moyennés. Les scores de chaque graine sont dans `results/results.json`.

## Lancer le code

```bash
pip install -r ../requirements.txt
```

Place le fichier `AMZN_2006-01-01_to_2018-01-01.csv` dans ce dossier, puis :

```bash
python rnn_amzn.py
```

Compter 10 à 25 minutes sur CPU. Pour un test rapide :

```bash
python rnn_amzn.py --lookbacks 10 --hidden-dims 32 --n-seeds 1 --epochs 5
```

Options : `--csv`, `--lookbacks`, `--hidden-dims`, `--epochs`, `--lr`, `--batch-size`, `--train-frac`, `--n-seeds`, `--seed`, `--out-dir`.

## Limites

- Un seul actif et une seule période de test (2015-2017, environ 600 jours) : l'erreur dépend beaucoup de quelques journées de forte variation.
- Hyperparamètres d'entraînement (époques, learning rate) non optimisés.
- Les écarts entre configurations sont petits : sans test statistique, on ne peut pas affirmer qu'une configuration est meilleure qu'une autre.
- Un seul signal en entrée (le prix de clôture) : pas de volume, d'indicateurs ni d'information externe.
