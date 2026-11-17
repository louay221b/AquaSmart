AquaSmart — Protocole temporel expérimental (version additive)
================================================================

OBJECTIF
- Découper chaque bassin par JOURS ENTIERS chronologiques (70/15/15 approx.),
  après filtrage, en conservant la même partition pour toutes les méthodes.
- Ajuster AUCUN scaler sur validation/test ou données pooled : calibration publique fixe.
- Sous-échantillonner SEULEMENT les données TRAIN et conserver VAL/TEST complets.
- Evaluer chaque modèle local sur le test de SON bassin; comparer les erreurs
  pooled et macro en utilisant la même partition temporelle.
- NE PAS ecraser les anciens résultats ni les scripts historiques.

INSTALLATION (dans CMD à la racine AquaSmart-v2-FL-SMPC)
1) Conserver la structure du ZIP : fl_smpc/temporal_protocol.py et
   fl_smpc/run_temporal_experiments.py sont à copier dans ton fl_smpc actuel.
2) .venv\Scripts\activate
3) python -m py_compile fl_smpc\temporal_protocol.py fl_smpc\run_temporal_experiments.py
4) python -m fl_smpc.run_temporal_experiments --sample-cap 3000 --rounds 10
5) Examiner fl_smpc\results\temporal_comparison_results.json

LIMITES SCIENTIFIQUES IMPORTANTES
* Les étiquettes de ration sont simulées par heuristique ; pas de ration réelle mesurée.
* IoTPond12 n'a que 9 jours avant filtre : le test risque d'être instable.
* Trier par jour local ne signifie pas convertir proprement les fuseaux UTC.
* La comparaison utilise un modèle linéaire à gradients, pas XGBoost.
* La validation est réservée mais pas encore utilisée pour l'optimisation.
* Le « masquage » fonctionne en un seul processus, et NE GARANTIT PAS les
  propriétés cryptographiques d'un protocole SMPC ; latence/traffic simulés.
* Avant publication : ablations des clients, seeds répétés, incertitudes,
  protocole SMPC réaliste, vérification unité capteurs et si possible vérité terrain.
