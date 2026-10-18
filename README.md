# Deep Learning - Lab 1 : MLP avec PyTorch

**Outils :** Kaggle Notebooks (CPU), Python, PyTorch, scikit-learn, skorch, pandas, seaborn, GitHub.

---

# Partie 1 : Régression (NYSE)

**Notebook :** [part1_regression_nyse.ipynb](part1_regression_nyse.ipynb)
**Dataset :** [New York Stock Exchange](https://www.kaggle.com/datasets/dgawlik/nyse), fichier `prices-split-adjusted.csv`

## Tâche
Prédire le prix de clôture `close` à partir de `open`, `low`, `high` et `log1p(volume)`.

## 1. Analyse exploratoire des données (EDA)
Le jeu de données contient **851 264 observations**, **7 variables** (date, symbol, open, close, low, high, volume) et **501 entreprises**, du **04/01/2010 au 30/12/2016**, sans valeur manquante. Les prix valent quelques dizaines à quelques centaines de dollars alors que le volume atteint des millions : une standardisation est nécessaire.

### Évolution des prix
![Prix](images/p1_prix_temps.png)

Les entreprises ont des niveaux de prix très différents et des tendances propres, avec une hausse globale sur la période. Cette hétérogénéité explique la forte dispersion de `close` (écart-type d'environ 93 $).

### Distributions
![Distributions](images/p1_distributions.png)

`close` et `volume` sont fortement asymétriques à droite. `log1p(volume)` est beaucoup plus symétrique, ce qui justifie la transformation logarithmique.

### Corrélations
![Corrélations](images/p1_correlation.png)

`open`, `high`, `low` et `close` sont corrélés à plus de 0,99, alors que le volume est faiblement corrélé aux prix. Le problème est donc presque linéaire.

### open vs close
![open vs close](images/p1_scatter.png)

Les points forment presque une droite : le prix de clôture est très proche du prix d'ouverture du même jour.

## 2. Préparation des données
- Découpage **par date** (série temporelle) : entraînement avant 2016 (**725 139 lignes**), test à partir de 2016 (**126 125 lignes**).
- `StandardScaler` ajusté **uniquement sur le train**, appliqué aux features et à la cible.
- Transformation `log1p` sur le volume.

## 3. Architecture du réseau
MLP **4 → 32 → 16 → 1** : activations ReLU, perte MSE, optimiseur Adam. Les options Dropout et BatchNorm sont activables pour la régularisation.

## 4. Recherche d'hyperparamètres (GridSearchCV + skorch)
Le modèle PyTorch est encapsulé avec `skorch.NeuralNetRegressor` pour être utilisé dans `GridSearchCV` (sous-échantillon de 50 000 lignes, 3 plis, 16 combinaisons).

| Hyperparamètre | Valeurs testées |
|---|---|
| Learning rate | 0.01, 0.001 |
| Optimiseur | Adam, SGD |
| Epochs | 20, 40 |
| Neurones (couche cachée) | 32, 64 |

Les 8 meilleures combinaisons (score = -MSE sur la cible standardisée) :

| lr | Optimiseur | Epochs | Neurones | Score moyen | Rang |
|---|---|---|---|---|---|
| 0.001 | Adam | 20 | 32 | -0.000212 | 1 |
| 0.001 | Adam | 40 | 32 | -0.000251 | 2 |
| 0.001 | Adam | 20 | 64 | -0.000272 | 3 |
| 0.001 | Adam | 40 | 64 | -0.000311 | 4 |
| 0.010 | Adam | 40 | 64 | -0.000353 | 5 |
| 0.010 | SGD | 40 | 64 | -0.000354 | 6 |
| 0.010 | SGD | 40 | 32 | -0.000371 | 7 |
| 0.010 | SGD | 20 | 64 | -0.000410 | 8 |

**Meilleure combinaison :** lr = 0.001, Adam, 20 epochs, 32 neurones (MSE standardisé de 0,0002, soit environ 0,02 % de la variance de `close`).

**Observations :** un learning rate faible (0.001) avec Adam est le plus efficace. Un réseau plus petit (32 neurones) et moins d'epochs (20) font mieux que 64 neurones et 40 epochs : le problème est simple, donc une capacité supplémentaire n'apporte rien. Les écarts entre combinaisons restent très faibles (de l'ordre de 10⁻⁴).

**Limite :** `cv=3` ne respecte pas l'ordre temporel ; `TimeSeriesSplit` serait plus rigoureux.

## 5. Courbes Loss/Epochs et R²/Epochs
En régression, l'accuracy n'est pas définie : elle est remplacée par le **R²**.

![Courbes baseline](images/p1_courbes_baseline.png)

**Interprétation :**
- La loss (MSE sur cible standardisée) est déjà très faible dès la première epoch (train 1,9·10⁻⁴, test 3,4·10⁻⁴) et continue de diminuer légèrement jusqu'à l'epoch 20 (train 7,0·10⁻⁵, test 1,2·10⁻⁴).
- Le R² final est de 0,99993 sur le train et de 0,99995 sur le test.
- La loss de test reste au-dessus de celle du train, mais l'écart est faible et la loss de test continue de baisser : **il n'y a pas de surapprentissage**.
- Le problème est quasi linéaire (`open`, `high` et `low` sont très proches de `close`), ce qui explique la convergence très rapide.

**Métriques du modèle de base (en dollars) :**

| Jeu | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Train | 0.6124 | 0.7826 | 0.4102 | 0.99987 |
| Test | 1.0577 | 1.0285 | 0.5248 | 0.99990 |

L'erreur moyenne est d'environ 0,52 $ sur le test, pour des prix dont l'écart-type est d'environ 93 $. Le RMSE est plus élevé sur le test que sur le train, mais le R² y est légèrement meilleur, car la variance de `close` est plus grande sur la période de test.

## 6. Régularisation et comparaison
Techniques testées : Dropout (0.2), L2 / weight decay (1e-4), BatchNorm, Early stopping (patience 5) et une combinaison (Dropout + L2 + Early stopping).

| Modèle | Epochs | Train RMSE | Test RMSE | Test MAE | Test R² | Écart (test - train RMSE) |
|---|---|---|---|---|---|---|
| Baseline | 20 | 0.5731 | 0.7607 | 0.3923 | 0.9999 | 0.1876 |
| Dropout 0.2 | 20 | 3.8910 | 6.2907 | 2.2444 | 0.9963 | 2.3997 |
| L2 (wd=1e-4) | 20 | 0.6872 | 0.9093 | 0.4685 | 0.9999 | 0.2221 |
| BatchNorm | 20 | 3.3942 | 5.1039 | 2.5296 | 0.9976 | 1.7097 |
| Early stopping | 20 | 0.5731 | 0.7607 | 0.3923 | 0.9999 | 0.1876 |
| Combiné | 6 | 4.6309 | 7.6423 | 2.3953 | 0.9945 | 3.0114 |

![Comparaison](images/p1_comparaison_reg.png)

*Remarque : la baseline de ce tableau (test RMSE 0,76) diffère légèrement de celle de la section 5 (test RMSE 1,03) car il s'agit d'un nouvel entraînement : l'initialisation et le mélange des mini-batchs introduisent une variabilité entre exécutions. Les comparaisons se font au sein du même tableau.*

**Analyse :**
- **Aucune technique n'améliore la baseline.** Le modèle de base ne sur-apprend pas, donc la régularisation n'a rien à corriger.
- **Dropout et BatchNorm dégradent fortement les résultats** (test RMSE de 6,29 et 5,10, contre 0,76). Le réseau est très petit (32 puis 16 neurones) et doit reproduire avec précision une relation presque linéaire. Éteindre 20 % des neurones ajoute un bruit qui l'empêche d'atteindre cette précision. Pour BatchNorm, l'écart entre les statistiques de batch (entraînement) et les statistiques cumulées (évaluation) peut aussi biaiser les prédictions. Ces deux modèles **sous-apprennent** : leur erreur est élevée sur le train aussi, donc l'« écart » plus grand n'indique pas du surapprentissage.
- **L2 (1e-4)** dégrade légèrement (RMSE 0,91 contre 0,76) : la pénalité tire les poids vers 0 alors que la relation à apprendre demande des poids précis.
- **Early stopping** a donné exactement les mêmes résultats que la baseline : la loss de test continuait de baisser, donc l'arrêt ne s'est jamais déclenché (20 epochs effectués).
- **Combiné** s'est arrêté à l'epoch 6 : le bruit du dropout rend la loss de test irrégulière, ce qui a déclenché l'arrêt trop tôt, d'où le pire résultat.
- En termes de R², toutes les variantes restent au-dessus de 0,994 : même les moins bonnes expliquent presque toute la variance.

## 7. Limites
- Tâche facile : prédire `close` à partir de `open/high/low` du même jour donne un R² très élevé et laisse peu de place à l'amélioration.
- GridSearch sur un sous-échantillon, avec des plis qui ne respectent pas l'ordre temporel.
- L'early stopping surveille la loss de test, ce qui n'est pas rigoureux ; il faudrait un jeu de validation séparé.
- Le test (année 2016) couvre une seule période de marché.
- Les résultats varient légèrement d'une exécution à l'autre (graine non fixée pour le DataLoader).
- Une tâche plus réaliste serait de prédire le cours du lendemain à partir des jours précédents, où la régularisation serait plus utile.

## 8. Synthèse personnelle
- J'ai appris à standardiser avec les statistiques du jeu d'entraînement uniquement et à découper par date pour éviter toute fuite d'information.
- J'ai appris à utiliser skorch pour intégrer un modèle PyTorch dans `GridSearchCV`, et j'ai mesuré le coût en temps d'une recherche d'hyperparamètres.
- J'ai appris à lire les courbes de loss et de R² pour détecter le surapprentissage, ici absent.
- J'ai constaté que la régularisation n'est pas toujours bénéfique : sur un problème presque linéaire avec beaucoup de données, Dropout et BatchNorm ont dégradé le modèle (sous-apprentissage). Il faut d'abord diagnostiquer le problème avant de choisir une technique.
- L'accuracy n'a pas de sens en régression : on utilise R², RMSE et MAE.

---

# Partie 2 : Classification multi-classes
*(à venir)*
