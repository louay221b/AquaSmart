# Correction 3 — personnalisation proximale de SecFed

Extension additive : décompresser à la racine `aina_pkg`, à côté de `correction_v1`, `correction_v2` et `sims`. Python et dépendances inchangés. Ne remplace aucun fichier ancien.

## Protocole
- Les partitions externes de correction 1 sont conservées (blocs de poids et temps). Les blocs sont des approximations, pas des identifiants de mesures vérifiés. Le test externe n'est ni entraîné ni évalué ; la constitution des partitions repose toutefois sur les blocs définis à partir du poids, comme dans correction 1.
- La partie train externe est divisée au voisinage de 70/30 par blocs indivisibles. Les hyperparamètres sont retenus sur cette validation interne, avec imputation et scaler refaits uniquement sur train interne. Après sélection, les modèles et le prétraitement sont réajustés sur train externe. La validation externe reste exploratoire, car déjà examinée dans les versions précédentes.
- FedAvg : grille alpha de rcore, learning rates 0.05/0.1, epochs 1/5, rounds 20/40/80. `--quick` utilise une grille réduite et n'est pas la campagne principale.
- Personnalisation analytique : minimiser MSE + alpha ||w||² + lambda ||theta-theta_global||², alpha>0; lambda>=0 ou modèle global gelé (JSON null). Le biais n'est pas pénalisé par Ridge mais est inclus dans la pénalité proximale. Lambda=0 correspond au Ridge local.
- Sélection commune ou par bassin si >=5 blocs internes de train ET validation; sinon configuration commune. Ce seuil est une décision exploratoire et ne garantit pas cinq mesures indépendantes. La baseline Ridge locale utilise la même règle de sélection locale/commune, pour comparaison équitable. Les baselines communes sont choisies selon le même critère global. Les valeurs sont sélectionnées par graine sur les seules données internes.
- Les masques gaussiens restent une simulation; les coûts sont théoriques, le scaler mutualisé n'est pas privé. Aucune preuve de confidentialité cryptographique ni de nouveauté algorithmique n'est revendiquée.
- Les résultats de correction 3 ne se comparent pas directement aux scores optimisés sur la validation externe de correction 2, puisque la procédure de sélection change. Comparer les modèles de correction 3 entre eux.

## CMD Windows (depuis aina_pkg, .venv activé)

```
python -m unittest discover -s correction_v3/tests -v
python correction_v3\run_adaptive.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode intervals --seeds 42 --quick
python correction_v3\run_adaptive.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode intervals --seeds 0,1,2,3,4
python correction_v3\run_adaptive.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode rows --seeds 0,1,2,3,4
```

Par défaut sélection macro. `--selection micro` disponible pour analyse de sensibilité, pas nécessaire pour le premier passage. Résultats `adaptive_results/intervals_macro_quick.json`, `intervals_macro.json`, `rows_macro.json`. Comparer RMSE micro/macro, R², RMSE par bassin, configuration choisie et équivalence numérique.

Les cinq tests synthétiques couvrent solution Ridge, stationnarité, frontières temporelles, fallback petits bassins, absence d'utilisation des cibles de validation externe pour choisir les paramètres, et équivalence des masques. Ils ne remplacent pas l'exécution sur les CSV réels. La variante n'est pas une implémentation de SMPC cryptographique.
