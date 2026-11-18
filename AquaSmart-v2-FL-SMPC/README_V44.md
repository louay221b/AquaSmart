# AquaSmart — V4.4 (tolérance limitée aux déconnexions + processus isolés)

Cette version est **additive**. Elle ne remplace aucune V2/V3/V4.1/V4.2/V4.3.

## Contenu de l'archive

```
fl_smpc/secure_aggregation_v44.py      # Crypto SecAgg + récupération avant upload
fl_smpc/process_secagg_v44.py         # Processus clients + serveur séparés
fl_smpc/run_temporal_experiments_v44.py  # Comparaison VAL uniquement
 tests/test_temporal_v44.py          # Huit tests
SECURITY_MODEL_V44.md                 # Hypothèses et limites de sécurité
README_V44.md                         # Instructions
```

Pré-requis : la V4.3 installée et ses dépendances Python ; aucune dépendance
supplémentaire au-delà de `requirements_v43.txt`.

## Installation — Windows CMD

1. Télécharger `AquaSmart_V4_4_Dropout_Process_SecAgg.zip` dans Téléchargements.
2. Depuis `C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC` :

```cmd
.venv\Scripts\activate
powershell -NoProfile -Command "Expand-Archive -LiteralPath (Join-Path $env:USERPROFILE 'Downloads\AquaSmart_V4_4_Dropout_Process_SecAgg.zip') -DestinationPath (Get-Location).Path -Force"
python -m pip install -r requirements_v43.txt
python -m py_compile fl_smpc\secure_aggregation_v44.py fl_smpc\process_secagg_v44.py fl_smpc\run_temporal_experiments_v44.py
python -m unittest discover -s tests -p "test_temporal_v44.py" -v
```

Les huit tests doivent afficher `OK`. Le test de processus utilise `spawn`
compatible Windows.

## Première expérience sur les 11 bassins (validation uniquement)

```cmd
python -m fl_smpc.run_temporal_experiments_v44 --sample-cap 3000 --rounds 2 --seeds 42 --dropouts IoTPond10
```

La commande compare, avec la même seed, un entraînement à participation
complète et un scénario où **IoTPond10 annonce sa clé mais n'envoie aucune
mise à jour**. Un minimum de 2 uploaders est requis. Les poids d'agrégation
sont renormalisés entre les clients qui ont effectivement envoyé une mise à
jour. Elle produit des fichiers JSON dans `fl_smpc/results/temporal_v44`.

Options : `--rounds 1`, `--dropouts IoTPond10,IoTPond11`, `--seeds 42,11`.
Lancer peu de rounds au départ car **chaque round démarre de nouveaux processus
OS**, ce qui crée un surcoût important sur Windows.

## Mesures disponibles

- Par scénario : `max_plain_secagg_param_diff`, RMSE micro/macro de **validation**,
  nombre de survivants et de déconnexions, durée de chaque round, octets et
  messages JSON réellement envoyés via `multiprocessing.Pipe`.
- Pas de sélection de modèles fondée sur TEST dans cette version.
- La métrique IPC inclut les messages de contrôle et d'agrégation, mais pas
  la transmission des vecteurs lors du **démarrage** des processus. Ce n'est
  pas du trafic réseau ni du temps réseau.

## Limites de sécurité importantes

V4.4 n'est pas un protocole SMPC à sécurité active. Hypothèses : clients
honnêtes, serveur honnête mais curieux, sans collusion. La récupération
fonctionne seulement pour les déconnexions **après annonce de clé et avant
upload**, si les uploaders restants terminent la phase de récupération.
Le serveur peut connaître la **somme des survivants**, mais ne doit pas recevoir
leurs mises à jour individuelles à son interface. L'orchestrateur du benchmark
conserve les vecteurs sources pour le test ; ce n'est pas un déploiement
privacy-preserving complet.

Référence technique : `SECURITY_MODEL_V44.md`.