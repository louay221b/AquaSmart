# SecFed correction V4 — évaluation rétrospective figée

Pré-requis : dossier `aina_pkg` avec `sims/rcore.py` et corrections V1, V2 et V3, même environnement Python et mêmes CSV que V3. Aucun ancien fichier remplacé.

`v3_reference.json` contient les configurations par seed et les RMSE de validation des deux campagnes V3 complètes, extraits de l'archive `SecFed_adaptive_results.zip` (sans la seed 42). Les empreintes SHA-256 des CSV et les comptes des partitions doivent correspondre. Le script réutilise les configurations V3, ne retune pas les méthodes historiques et vérifie leurs RMSE micro, macro et par bassin à 1e-7 près AVANT tout scoring test. Les deux nouvelles ablations (Local/Global binaire et SGD) sont sélectionnées uniquement sur la partition interne de V3, à partir de sa configuration globale figée; la sélection est macro. La sélection binaire utilise la même grille alpha, les mêmes seuils d'éligibilité et la même règle de repli que V3, sans lambda fini positif. La personnalisation SGD utilise une configuration commune. Aucun réentraînement sur train+validation.

## Commandes depuis aina_pkg, Windows CMD

```cmd
python -m unittest discover -s correction_v4/tests -v
python correction_v4\run_frozen.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode intervals --phase freeze
python correction_v4\run_frozen.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode rows --phase freeze
```

Archiver les deux manifestes avant de lancer le test. Le script affiche leurs SHA-256. Les manifestes contiennent les configurations et l'audit inter-bassins. Ils ne sont ni signés ni inviolables : conserver une copie datée et ne pas les modifier. Si le contrôle de reproduction échoue, ne pas contourner le contrôle; examiner les versions et les données.

```cmd
python correction_v4\run_frozen.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode intervals --phase test --accept-retrospective-test
python correction_v4\run_frozen.py --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --mode rows --phase test --accept-retrospective-test
powershell -NoProfile -Command "Compress-Archive -Path 'frozen_results\*.json' -DestinationPath 'SecFed_frozen_results.zip' -Force"
```

Les deux phases refusent d'écraser leurs sorties. Le mode test exige le manifeste et contrôle les empreintes du code et des données. Il affiche les RMSE micro et macro, enregistre R², RMSE par bassin, nombre d'unités de test, erreurs numériques et coûts théoriques.

## Limites

- Intervals = analyse principale, rows = sensibilité. Les séries de poids contiguës sont des proxies des mesures, sans identifiant de mesure manuel.
- La lecture et le découpage des CSV utilisent l'ensemble des timestamps/poids pour identifier les blocs. Aucun score test ni sélection sur les cibles test dans freeze. Test uniquement pour scorer dans la phase test; transformations ajustées sur train.
- Fenêtres par bassin non synchronisées, donc évaluation rétrospective et non prospective; audit des dates fourni, aucun nouveau split global dans V4.
- Les tests ont déjà été examinés dans le manuscrit original avec un autre protocole; ils ne sont pas totalement inédits. Seeds non indépendantes temporellement.
- Masques gaussiens simulés, scaler global calculé en clair; aucune nouvelle garantie cryptographique. Pas encore de test sur le dataset énergétique.
- Sept tests synthétiques passent; reproduction des résultats réels à vérifier sous Windows.
