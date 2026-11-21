# SecFed — correction scientifique, étape 1

Cette archive est extraite des sources de l'encadrante. Les sources et résultats historiques sont conservés. Le nouveau code est dans `correction_v1/`, le manuscrit révisé dans `secfed_aina_medium_correction_v1.tex`. Ne pas soumettre le manuscrit avant le recalcul des résultats et figures.

## Corrections du protocole
- Relecture des CSV bruts sans le cache pickle historique (empreintes SHA-256 dans l'audit).
- Même cible log(1+poids en grammes), six capteurs et modèle Ridge/mini-batch SGD que le manuscrit. Seuils de nettoyage conservés autant que possible, nitrate borné avant imputation, valeurs non finies éliminées, formats de date explicites.
- Découpage chronologique approximativement 60/20/20 en nombre de lignes, sans couper un bloc de poids constant contigu ou un timestamp identique. Les blocs sont calculés avant le filtrage des capteurs. Ils sont un proxy des intervalles de mesure (des mesures successives peuvent avoir le même poids). Ce n'est pas un découpage par identifiant réel de mesure ni un holdout véritablement inédit. Les labels historiques servent à déterminer ces frontières rétrospectives.
- Au moins 200 lignes et cinq blocs par bassin; exclusions documentées dans l'audit. Ce seuil de cinq est un minimum technique, pas une garantie statistique. Les dates seules sont acceptées mais ne permettent pas de distinguer les heures. Les suffixes CET/CEST/UTC sont retirés sans conversion; les règles de datation doivent être vérifiées avant la campagne définitive.
- Imputation du nitrate: médiane des seules lignes d'entraînement effectivement sélectionnées, par bassin; secours par médiane globale d'entraînement. Aucun remplissage à partir de la validation/test. Échec explicite si tout le nitrate d'entraînement manque. Standardisation sur entraînement uniquement. Le scaler mutualisé n'est pas privé.
- Deux modes: lignes (`rows`) et une ligne par série de poids constant (`intervals`, moyenne des capteurs). Mêmes frontières avant agrégation. Tous les bassins techniquement éligibles sont conservés; ne pas confondre cette variante avec l'ancienne analyse 9 bassins >=20 intervalles.
- Sélection alpha sur validation; scores de validation exclusivement. Aucune évaluation de test par le runner, mais préparation rétrospective des partitions à partir du CSV complet. Le test historique n'est pas indépendant au sens confirmatoire. Pas encore de personnalisation.
- Baselines: mean pooled, own mean, local ridge, centralized ridge, FedAvg et masques gaussiens simulés. Budgets, grille, taux d'apprentissage et epochs de l'article conservés. Compteurs d'octets théoriques, pas mesures réseau. Pas de cryptographie réelle dans ce runner.

## Windows CMD
Extraire cette archive dans `C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\secfed_review` (elle contient un dossier `aina_pkg`). Depuis ce dossier `aina_pkg`, avec le venv AquaSmart activé:

```
python -m unittest discover -s correction_v1/tests -v
python correction_v1\run_corrected.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode audit
```

Examiner `correction_results/audit.json` avant de lancer l'entraînement. Si le nombre de bassins ou les tailles diffèrent, ne pas forcer les résultats. Envoyer l'audit pour vérifier les exclusions et la distribution des intervalles. Ensuite seulement:

```
python correction_v1\run_corrected.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode rows --seeds 0,1,2,3,4
python correction_v1\run_corrected.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode intervals --seeds 0,1,2,3,4
```

Dépendances du nouveau runner: numpy, pandas; le module historique `sims/rcore.py` est importé pour les modèles, pas pour charger les CSV. Les autres scripts historiques contiennent encore des chemins absolus; ne pas les lancer comme s'ils avaient été portés.

## Manuscrit
Corrections rédactionnelles en anglais: 64x plain FL contre 37x Gaussian SecFed; 1.2% masquage simulé uniquement; coût idéal et timings du prototype distingués; garantie gaussienne conditionnelle et hypothèses fixes du corollaire; limites des graphes déconnectés, du DP et du prototype crypto; résultats hérités marqués préliminaires. Les figures historiques sont conservées mais leurs textes internes doivent être régénérés. Bibliographie et références non vérifiées systématiquement. Compilation LaTeX première passe réussie, mais BibTeX absent de l'environnement: compiler entièrement sur Overleaf avec `secfed_aina_medium_correction_v1.tex` comme document principal. Aucun PDF final fourni.

## Validation
5 tests synthétiques réussis: frontières de groupes, timestamps identiques, invariance du preprocessing d'entraînement à une modification future du nitrate, mode intervalles, invariance des scores de validation à une modification des cibles test après préparation. Les CSV réels ne sont pas inclus dans l'archive originale et n'ont pas été exécutés ici. Pas de modification du dépôt GitHub.

## Étapes restantes
Valider l'audit réel; relancer les modèles et les analyses dépendantes (crypto, réseau, LOPO, DP, convergence, sensibilité); harmoniser le comparatif énergétique; ajouter la personnalisation sur nouvelles cibles après fixation du protocole; choisir une évaluation temporelle finale honnête (nouvelles données si possible); comparer les coûts sur le même périmètre; refaire tables/figures et vérifier les références. Ne pas présenter les anciennes valeurs comme nouveaux résultats.
