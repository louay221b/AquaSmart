AQUASMART - V4.1 | Optimisation FedAvg sur VALIDATION
======================================================

But
---
Comparer Centralized, Local, FedAvg de base, FedAvg optimise et FedAvg
optimise avec simulation de masquage additif, sur les memes partitions
chronologiques. Aucun hyperparametre n'est choisi a partir du test.

Aucune modification des fichiers V2, V3 ou des CSV.

Installation (Windows CMD)
--------------------------
1. Extraire ce ZIP dans :
   C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC
   Fusionner les dossiers fl_smpc et tests, sans remplacer l'ancien contenu.

2. Structure attendue :
   AquaSmart-v2-FL-SMPC/
     fl_smpc/
       temporal_protocol.py                (deja installe - V2)
       run_temporal_experiments_v3.py      (deja installe - V3)
       run_temporal_experiments_v4.py      (NOUVEAU)
     tests/
       test_temporal_v4.py                (NOUVEAU)

3. Dans CMD :
   cd /d C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC
   .venv\Scripts\activate
   python -m py_compile fl_smpc\run_temporal_experiments_v4.py
   python -m unittest discover -s tests -p test_temporal_v4.py -v

4. Essai rapide (8 combinaisons, trois checkpoints) :
   python -m fl_smpc.run_temporal_experiments_v4 --sample-cap 3000 --rounds 5,10,20 --lr-grid 0.005,0.01 --alpha-grid 0.1,1 --epochs-grid 1,5 --seeds 42 --selection macro_rmse

5. Grille approfondie (18 combinaisons, quatre checkpoints) :
   python -m fl_smpc.run_temporal_experiments_v4 --sample-cap 3000 --rounds 5,10,20,50 --lr-grid 0.005,0.01,0.02 --alpha-grid 0,0.1,1 --epochs-grid 1,5 --seeds 42 --selection macro_rmse

Sorties
-------
fl_smpc/results/temporal_v4/seed42_*.json
fl_smpc/results/temporal_v4/summary_*.json
Un identifiant stable associe a la grille evite de remplacer les autres runs.
La sortie inclut les 18 resultats de validation, les parametres choisis, la
prediction de test exploratoire, les scores par bassin, les couts estimes.

Methode
-------
- Partition par jour, Train/Val/Test, avec cap uniquement sur Train (V2).
- Baselines V3 avec lr=.01, alpha=1.0, local_epochs=5.
- FedAvg optimise : recherche exhaustive de (lr, alpha, local_epochs)
  et nombre de rounds, en minimisant macro-RMSE sur VAL.
- Rejeu de la version masquee avec les MEMES parametres, rounds et seed.
- Test : calcule apres selection; PAS UTILISE pour selectionner.
- 8 et 18 observations de validation pour IoTPond10/11 : prudence.

IMPORTANT POUR L'ARTICLE
------------------------
- La cible feed_ration_g est simulee, pas une ration reellement mesuree.
- Le masquage est une emulation numerique sans SMPC cryptographique.
- Les tests temporels de V2/V3 ont deja ete regardes : ceux-ci restent
  des resultats EXPLORATOIRES, pas un test final aveugle.
- Les couts reseau sont des estimations, et non des mesures reelles.
- Les budgets de calcul central/local/FL ne sont pas strictement egalises.
- Les dix seeds seules ne remplacent pas plusieurs fenetres temporelles.
- La V4.1 NE comprend PAS encore Personalized FL ni le SMPC cryptographique.

Une etape ulterieure doit definir de nouvelles fenetres temporelles non
consultees durant la conception des methodes, puis fixer la grille et
l'algorithme avant evaluation finale.
