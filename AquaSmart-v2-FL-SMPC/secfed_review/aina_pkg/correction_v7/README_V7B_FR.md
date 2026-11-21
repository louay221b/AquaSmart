# V7B — Entraînement récursif avec agrégation TLS (prototype)

Extension additive de V7A. Conserver V1–V7A, les CSV et les fichiers JSON de test V5/V6 exactement tels quels. Le script vérifie les empreintes des manifestes et du code V5/V6 via V7A, et l'empreinte du JSON de test antérieur pour les expériences complètes sans dropout. Ne pas modifier les anciens scripts. Le fichier JSON de test V5 attendu est `synchronized_results/intervals_test.json` ; celui de V6 est `bdg2_results/bdg2_test.json`.

## Windows CMD (depuis aina_pkg, venv activé)

`python -m pip install psutil` (facultatif, nécessaire seulement pour les mesures RSS).

```
python -m unittest discover -s correction_v7/tests -v
python correction_v7\run_crypto_full.py --domain aqua --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --smoke
python correction_v7\run_crypto_full.py --domain bdg2 --dataset-dir bdg2_dataset --smoke
python correction_v7\run_crypto_full.py --domain aqua --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --seeds 0 --accept-retrospective-test
python correction_v7\run_crypto_full.py --domain bdg2 --dataset-dir bdg2_dataset --seeds 0 --accept-retrospective-test
```

Smoke = 2 rounds, aucun score test. Campagnes complètes : même commandes sans `--smoke`, seeds restantes `--seeds 1,2,3,4`. Les 5 seeds représentent 400 rounds Aqua et 180 rounds BDG2. Chaque round crée de nouveaux processus et une PKI de test ; ne pas extrapoler ces durées à une architecture persistante. Un fichier par seed/scénario, refus d'écrasement. Le scénario facultatif `--dropout` déclare le premier client alphabétique absent avant upload à chaque round (calcul local toujours effectué, tous les sites évalués après apprentissage), avec une référence plaintext appariée sur le même ensemble actif et renormalisation des poids ; non comparable à un dropout aléatoire. Il n'est pas nécessaire pour la première campagne complète.

```
powershell -NoProfile -Command "Compress-Archive -Path 'crypto_full_results\*.json' -DestinationPath 'SecFed_crypto_full_results.zip' -Force"
```

## Méthode et mesures

La sortie décodée de la somme TLS est réellement réinjectée dans le round suivant. La référence plaintext et TLS utilisent le même ordre et les mêmes permutations RNG. Sans dropout, la trajectoire plaintext est comparée à r.fedavg (tolérance 1e-10), et les scores globaux et SGD aux anciens résultats figés (tolérance 1e-7, incluant les RMSE par site et R²). Les deux trajectoires appliquent ensuite la même adaptation SGD figée. Le test n'intervient pas dans l'ajustement. La somme de chaque round est comparée au plaintext des contributions issues de la trajectoire TLS, garde K_actifs/(2S)+1e-10 ; la divergence entre trajectoires est mesurée séparément. Aucune borne de convergence globale déduite automatiquement de la borne de quantification. Pas de modèles ou contributions enregistrés, seulement métriques et configurations.

Les octets mesurés sont les trames applicatives JSON (envoi + réception) de la somme sécurisée. Ne comprennent ni TLS handshake, ni TCP/IP, ni le broadcast vers les processus d'entraînement (contrôleur central), ni les échanges de prétraitement. `ideal_plain_model_payload_bytes` est un modèle théorique, pas un trafic réellement mesuré. Le chronométrage `tls_call_wall_sec` inclut le lancement/arrêt des processus, PKI et agrégation. Les durées d'entraînement excluent la préparation des données, mais incluent l'échantillonnage mémoire facultatif. RSS : somme échantillonnée des RSS du parent et des descendants toutes les 50 ms ; les pages partagées sont comptées plusieurs fois et des pics brefs peuvent être manqués. Ce n'est ni une consommation RAM physique exacte ni une mesure isolée des primitives. Les percentiles concernent les rounds de la même exécution, pas des répétitions indépendantes.

Limites : contrôleur de confiance voit toutes les données et mises à jour ; calcul local simulé dans le contrôleur, communication réelle entre workers sur localhost, certificats éphémères, serveur et clients honnêtes non collusifs. Pas de partage de secret à seuil ni de sécurité prouvée face à un serveur malveillant. Les jeux de test rétrospectifs ont déjà été examinés. La validation synthétique ne garantit pas le bon fonctionnement sur les CSV réels Windows.
