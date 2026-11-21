# V6B — BDG2, évaluation à calendrier commun

Extension additive : nécessite les corrections V1 à V5 et la V6A déjà installées. Les fichiers originaux ne sont pas remplacés. Extraire le ZIP dans `aina_pkg`.

## Protocole figé
- Audit V6A obligatoire : `bdg2_results/calendar_audit.json`; vérification des tailles, SHA-256 et nombres de lignes admissibles. Les timestamps électriques doivent être strictement croissants.
- Cohorte définie par l'audit, sans sélection sur les scores de test. Calendrier V6A, raw electricity strictement positive, jointure météo site/timestamp, surface valide. Sept variables originales, y=log1p(kWh), huit paramètres avec biais.
- Échantillonnage uniforme sans remise de 3 000 heures-bâtiments par site et seed pour l'entraînement interne, 10 000 pour la validation (même échantillon de validation pour toutes les seeds). Après sélection, nouvel échantillon de 3 000 heures-bâtiments par site dans train+validation pour le réentraînement. Le test est intégral, pas échantillonné.
- Normalisation globale ajustée sur le seul échantillon d'entraînement de chaque phase; écart-type inférieur à 1e-8 remplacé par 1. Pas de confidentialité du prétraitement. Les jours valides remplacent les blocs de poids pour l'éligibilité à l'adaptation locale. Grilles et méthodes V5; macro comme critère de sélection. L'agrégation pondère les tailles des échantillons (égales si tous les sites atteignent le cap).
- Pour le test, on accumule par site la matrice M=Σzzᵀ, z=[x brut,1,y]. Pour les coefficients remis à l'échelle brute, v=[w brut,b brut,-1], SSE=vᵀMv. Tous les modèles sont linéaires. RMSE micro à partir de la somme des SSE, macro moyenne des RMSE site, R² global. C'est équivalent au calcul ligne par ligne à l'arrondi numérique près. Pas besoin de mémoriser des millions de prédictions. Les statistiques du test ne sont pas utilisées pour entraîner ou sélectionner; la phase freeze ne calcule pas les moments du test (seulement les comptages de validité de l'audit).
- Masques gaussiens simulés seulement, coûts théoriques de communication hors prétraitement/tuning. Pas de prévision future à horizon donné; estimation contemporaine, données historiques déjà étudiées. Les timestamps sont ceux enregistrés et l'embargo ne prouve pas l'alignement UTC. L'exclusion des zéros peut biaiser le dataset. Les seeds ne donnent pas de fenêtres indépendantes.

## CMD Windows depuis aina_pkg, venv active
```
python -m unittest discover -s correction_v6/tests -v
python correction_v6\run_bdg2.py --dataset-dir bdg2_dataset --phase freeze --quick
python correction_v6\run_bdg2.py --dataset-dir bdg2_dataset --phase freeze
```
Le quick crée `bdg2_manifest_quick.json` et ne peut pas être utilisé pour un test officiel. La campagne complète crée `bdg2_manifest.json` et affiche son SHA256. Conserver le manifeste et transmettre sa sortie avant test. Ensuite, après vérification :
```
python correction_v6\run_bdg2.py --dataset-dir bdg2_dataset --phase test --accept-retrospective-test --manifest-sha256 EMPREINTE_DU_MANIFESTE_COMPLET
```
Le script refuse l'écrasement des résultats. Les six tests (dont trois V6A) ont réussi sur données synthétiques. CSV réels non disponibles dans l'environnement de développement.
