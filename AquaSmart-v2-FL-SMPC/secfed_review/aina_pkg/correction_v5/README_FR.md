# V5 — évaluation rétrospective à calendrier commun

Extension additive à installer dans `aina_pkg` à côté de correction_v1...v4 et sims. Ne modifie pas V1–V4. Même CSV et empreintes que V3; pas de téléchargement de données. Les résultats V4 restent inchangés.

## Protocole fixé avant calcul
- Entraînement interne : timestamps enregistrés avant le 20 juillet 2021 00:00.
- Validation interne : du 20 juillet inclus au 10 août 2021 exclu.
- Test : à partir du 10 août 2021 00:00 inclus.
- Les blocs V1 qui traversent les frontières sont entièrement écartés. Les timestamps sont des heures locales telles qu'enregistrées, sans garantie de conversion UTC. Les intervalles sont des proxys construits à partir des poids répétés, pas des mesures indépendantes vérifiées.
- Admission : au moins 5 blocs fit, 2 blocs de validation interne, 1 bloc test; possibilité de moins de 11 sites. Minimum 2 sites pour entraîner. Cette sélection par couverture est rétrospective; aucune sélection sur les scores de test. Le seuil individuel de personnalisation reste 5 blocs dans chacune des deux partitions internes. Les petits sites admissibles utilisent la configuration commune. Les dates et seuils constituent un choix méthodologique exploratoire à auditer, pas un optimum empirique.
- Même objectifs, grilles V3/V4 et critère macro. Réestimation des hyperparamètres uniquement sur la nouvelle validation interne; réutiliser les hyperparamètres V4 aurait pu incorporer des données postérieures à la nouvelle frontière. Réentraînement avec fit + validation interne (blocs complets) et prétraitement ajusté uniquement sur cette période. Cap 3000, seeds 0–4. Intervals principal, rows complémentaire.
- Attention : moyennes de capteurs sur les mêmes intervalles que le poids, délimitation par étiquette rétrospective. Ce n'est pas une prévision à horizon futur. Pas de données externes inédites, ni garantie de disponibilité réelle des étiquettes. Prétraitement centralisé et masquage gaussien simulé. Les coûts restent théoriques. Comparaisons V4/V5 descriptives, cohortes, partitions et quantité d'entraînement potentiellement différentes.

## Windows CMD (depuis aina_pkg, venv activé)
```cmd
python -m unittest discover -s correction_v5/tests -v
python correction_v5\run_synchronized.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --phase audit
```
Envoyer `synchronized_results/calendar_audit.json` ou le journal avant la phase suivante. Si couverture insuffisante, documenter le blocage; ne pas rechercher des frontières qui maximisent les performances.

## Après validation de l'audit
```cmd
python correction_v5\run_synchronized.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode intervals --phase freeze
python correction_v5\run_synchronized.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode rows --phase freeze
```
Archiver les manifestes et leurs SHA256 affichés. La commande de test exige la valeur SHA256 exacte du manifeste correspondant, à remplacer dans l'exemple :
```cmd
python correction_v5\run_synchronized.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode intervals --phase test --accept-retrospective-test --manifest-sha256 HASH_INTERVALS
```
Idem rows avec HASH_ROWS. Les sorties ne sont pas écrasées. Pour une exécution distincte, choisir un autre --output-dir et documenter pourquoi. Ne pas retoucher les configurations après lecture du test. Les tests synthétiques couvrent la purge, l'ordre intersites, l'exclusion pour couverture, les égalités de date et la sélection indépendante des cibles test sur partitions fixées. Le dataset réel n'a pas été exécuté ici.
