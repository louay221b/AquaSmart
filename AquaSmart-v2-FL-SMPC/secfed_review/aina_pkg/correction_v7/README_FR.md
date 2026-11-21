# SecFed V7A — Pilot cryptographique sur contributions figées

Cette extension est additive. Aucun fichier V1–V6 n'est modifié. Exécuter depuis `aina_pkg` dans le même environnement Python que les campagnes précédentes. Dépendances: numpy, pandas, cryptography (déjà utilisé dans AquaSmart V4.3–V4.5). En cas de `ModuleNotFoundError: cryptography`, installer avec `python -m pip install cryptography`.

## Objet et procédure

Les deux manifestes complets sont verrouillés sur les SHA256 fournis dans les journaux V5/V6. Le code de préparation et les configurations existantes sont vérifiés avant le calcul. Les CSV et les partitions restent identiques. Aucun nouvel hyperparamètre sélectionné. La campagne utilise par défaut seed 0, premier round (option `--checkpoint last` pour le dernier round après replay **en clair**). Le modèle global en clair est reproduit avec la même suite RNG; les contributions pondérées de ce checkpoint sont traitées par le prototype V4.5 inchangé (bibliothèques incluses dans `vendor/fl_smpc`). Le script ne fait pas encore un entraînement cryptographique de bout en bout. Il ne produit aucun score prédictif de test. Le chargeur BDG2 V6 calcule aussi les moments de test, ici ignorés; il ne réutilise pas les cibles de test pour l'entraînement.

Commandes CMD :

```
python -m unittest discover -s correction_v7/tests -v
python correction_v7\run_crypto_pilot.py --domain aqua --dataset-dir "C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\dataset" --dropout
python correction_v7\run_crypto_pilot.py --domain bdg2 --dataset-dir bdg2_dataset --dropout
```

Option `--dropout` réalise un round tous clients en ligne, puis un round indépendant où le premier client alphabétique se déconnecte avant l'upload. Les mêmes contributions calculées en clair sont réutilisées, sans réentraîner. L'agrégat attendu est la somme des contributions des survivants. Le contrôleur renormalise par leur masse de poids uniquement pour comparer les modèles moyens correspondants; ce n'est pas une simulation d'une trajectoire FL après dropout. Le script refuse d'écraser un résultat existant. Augmenter `--timeout 60` si le lancement de nombreux processus est lent. Les résultats JSON sont dans `crypto_pilot_results/`.

```
powershell -NoProfile -Command "Compress-Archive -Path 'crypto_pilot_results\*.json' -DestinationPath 'SecFed_crypto_pilot_results.zip' -Force"
```

## Mesures et limitations

- Maximum de l'erreur de somme par coordonnée par rapport à la somme en clair du même ensemble de clients. Quantification: arrondi à l'entier avec échelle 10^8, erreur théorique <= K_actifs/(2*10^8), plus tolérance numérique 10^-10 dans le test. Le rapport contient aussi l'erreur après renormalisation par la masse de poids.
- Application frames JSON avec préfixes de longueur sur TCP loopback/mTLS 1.3: compte les deux sens côté clients, exclut les handshakes TLS, les en-têtes TCP/IP et le trafic réseau physique. Le temps total inclut création des certificats éphémères et des processus. Le temps de somme NumPy en clair est une référence arithmétique seulement, pas un baseline réseau équivalent.
- Les contributions ne sont pas exportées dans les JSON. Le contrôleur de benchmark connaît cependant toutes les données et contributions individuelles. Ce n'est PAS une garantie de confidentialité contre le contrôleur, ni un déploiement multi-machines.
- Le prototype reprend le modèle de menace V4.5: serveur honest-but-curious, clients honnêtes non collusifs, identités mTLS authentifiées. Pas de résistance au serveur malveillant, pas de preuve de protection contre collusion, pas de récupération par partage de secrets à seuil. Dropout après upload non pris en charge (abandon du round).
- Les résultats du pilote ne démontrent pas l'équivalence prédictive sur une trajectoire sécurisée complète. La validation de la trajectoire et la personnalisation après agrégation cryptographique, les mesures mémoire et une référence réseau comparable sont les étapes suivantes.
- Les deux tests locaux couvrent la reproduction du checkpoint sur données synthétiques et le vrai transport loopback tous clients en ligne/avec dropout. Les CSV réels ne sont pas disponibles dans l'environnement de développement.
