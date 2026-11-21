# SecFed V6 — audit BDG2 (6A)

Cette extension ne modifie pas V1–V5. Elle audite les CSV réels avant tout entraînement. Elle n'inclut pas encore le runner de modèles.

Source officielle : https://github.com/buds-lab/building-data-genome-project-2 ; Miller et al., Scientific Data (2020), DOI 10.1038/s41597-020-00712-x.

Extraire de la même révision / release les fichiers data/metadata/metadata.csv, data/weather/weather.csv, data/meters/raw/electricity.csv. Les placer ensemble dans un dossier bdg2_dataset. Vérifier qu'ils sont les vrais CSV et non des pointeurs Git LFS. Conserver les fichiers originaux et noter la version téléchargée. Le choix raw est explicite ; le script original ne précisait pas sa provenance. Le choix kWh strictement positifs reproduit son filtre (biais possible).

Commandes depuis aina_pkg avec .venv actif :

    python -m unittest discover -s correction_v6/tests -v
    python correction_v6\audit_bdg2.py --dataset-dir bdg2_dataset

Sortie : bdg2_results/calendar_audit.json. Pas d'écrasement ; `--output autre.json` pour un nouvel audit. Le programme lit les compteurs par tranches de 168 timestamps (option --chunk-size), contrôle SHA256, les clés uniques et la couverture par site. Aucun entraînement, aucune sélection par RMSE.

Calendrier proposé avant expérience : 2016-01-01 <= fit < 2016-12-30 ; 2017-01-01 <= validation < 2017-06-29 ; 2017-07-01 <= test < 2018-01-01. Les deux jours précédant chaque nouvelle partition sont exclus. Dates locales enregistrées ; la marge réduit le risque de chevauchement des fuseaux, mais ne prouve pas leur exactitude. Audit des valeurs de timezone présentes dans les métadonnées. Cohorte : au moins 10 bâtiments correspondants et 30 jours avec des données valides dans chacune des deux partitions fit/validation, ainsi qu'au moins 10 bâtiments représentés dans chaque partition fit/validation. La couverture test est signalée séparément, jamais utilisée pour sélectionner par performance. Les compteurs négatifs, nuls ou non finis et les observations sans météo finie sont écartés. Aucun remplissage. Les doublons temporels ou météo interrompent l'audit au lieu d'être corrigés arbitrairement.

Cible prévue log1p(kWh), sept caractéristiques originelles : température, point de rosée, vitesse du vent, heure sin/cos, weekend, log surface. C'est une estimation de consommation contemporaine, pas une prévision à horizon futur. Les modèles, normalisation sur fit et sélection interne viendront après vérification de l'audit. Ne pas comparer les RMSE numériques de cibles poids et électricité.

Tests synthétiques : 3 tests passent. Aucun CSV BDG2 réel fourni à ce stade. Pas de publication GitHub.
