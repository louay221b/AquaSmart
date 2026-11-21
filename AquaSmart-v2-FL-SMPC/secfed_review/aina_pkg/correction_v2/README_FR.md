# SecFed — correction étape 2 (personnalisation)

Extension additive à extraire dans `secfed_review/aina_pkg` (ne pas remplacer correction_v1 ni sims). Même préparation et mêmes CSV que l'étape 1. Aucune nouvelle dépendance.

## Windows CMD, environnement .venv déjà activé

```
python -m unittest discover -s correction_v2/tests -v
python correction_v2\run_personalized.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode intervals --seeds 42 --quick
python correction_v2\run_personalized.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode rows --seeds 0,1,2,3,4
python correction_v2\run_personalized.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode intervals --seeds 0,1,2,3,4
```

--quick est uniquement un test d'intégration. Les campagnes complètes sélectionnent le RMSE micro (même critère que l'étape 1). L'alternative --selection macro sert à une analyse de sensibilité à déclarer séparément : les baselines de l'étape 1 restent sélectionnées en micro. Les résultats sont sauvegardés après chaque seed dans `personalized_results/rows_micro.json` et `personalized_results/intervals_micro.json` (ou `intervals_micro_quick.json`). Ne pas mélanger le pilote et les répétitions 0–4.

## Méthodologie

- Baselines phase 1 recalculées sur les mêmes partitions et plafonds ; FedAvg initial 20 rounds, 5 epochs, lr .05, alpha choisi dans la grille originale.
- FedAvg optimisé : mêmes 7 alpha, lr .01/.05/.1, 1/5 epochs, checkpoints 20/40 rounds. 42 trajectoires, 84 candidats. Sélection sur validation. SecFed gaussien réexécuté avec la configuration choisie, mêmes seeds d'entraînement.
- Personnalisation : même alpha que le modèle global, fine-tuning sur train local uniquement, 0/1/3/5/10/20 epochs, lr .005/.01/.05. Une configuration commune, pas de choix optimiste différent pour chaque bassin. 0 epoch = pas d'adaptation ; 16 candidats distincts. Les hyperparamètres sont sélectionnés sur les fenêtres de validation, séparément pour chaque seed. Les paramètres personnalisés ne sont pas réagrégés dans le scénario modélisé.
- Local SGD témoin : initialisation à zéro, même alpha, ordre stochastique et budget d'adaptation que la méthode personnalisée. Contrôle de l'initialisation, pas un budget d'entraînement total identique. Local Ridge analytique reste le comparateur local principal.
- Erreurs globales et personnalisées entre masques gaussiens et FedAvg contrôlées à 1e-8. Coûts de communication du modèle théorique uniquement, pas de réseau. Le prétraitement partagé (scaler) n'est pas privé, le masquage gaussien est une simulation. Aucune nouvelle garantie de sécurité cryptographique.
- Résultats de validation exploratoires, sélection et évaluation sur les mêmes données ; les candidats sélectionnés incluent l'absence d'adaptation, donc une amélioration du critère de sélection sur validation n'est pas une preuve de généralisation. Les seeds partagent les fenêtres et ne sont pas des tests indépendants. Aucun score de test calculé. Les partitions de test existent toujours dans le prétraitement de phase 1, mais ne sont pas passées aux algorithmes de phase 2.
- Trois tests synthétiques locaux réussis sous Linux ; pas encore exécuté sur le dataset réel ni sous Windows.
