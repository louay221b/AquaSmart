# SecFed V8A — benchmark apparié des coûts

À extraire dans `aina_pkg`, en conservant V7A/V7B et leurs fichiers. Dépendances existantes : numpy, cryptography, psutil (Python 3.12 sur le PC). Aucun CSV utilisé. Aucun code V7 modifié sur disque.

```
python -m unittest discover -s correction_v8/tests -v
python correction_v8\run_cost_benchmark.py --smoke --output cost_results\v8a_smoke.json
python correction_v8\run_cost_benchmark.py --repeats 5
```

La campagne complète fait 25 paires (50 appels), K=4,8,15,20 à dimension fixe 8, puis K=8,d=7. Même vecteur synthétique pour les deux variantes, ordre alterné. La référence mTLS sans masquage reçoit une contribution et retourne la somme (deux trames par client), le protocole sécurisé V7 utilise six trames. Même fonction d'orchestration, PKI éphémère, processus spawn, délais, connexions, encodage JSON et instrumentation. Les cibles des processus sont substituées temporairement dans le parent pour la référence seulement; le worker serveur substitue son gestionnaire de connexions dans son processus. Pas de modification de fichier V7. Ne pas appeler cette référence en parallèle depuis plusieurs threads du parent.

Résultat `cost_results/v8a_benchmark.json` sauvegardé progressivement après chaque essai, avec `complete=true` seulement si toutes les paires sont terminées. En cas d'erreur, préserver le fichier incomplet et choisir `--output` différent pour retenter. Ce benchmark mesure le coût total de deux protocoles d'agrégation mTLS, pas le coût isolé des primitives ni le trafic réseau brut. Les volumes comprennent les trames JSON et leur préfixe, pas les handshakes, les en-têtes ou le transfert du modèle aux processus d'entraînement. Les RSS sommés peuvent compter les pages partagées plusieurs fois. Les contributions synthétiques ne valident pas l'utilité prédictive, déjà étudiée V7B. Démarrage des processus à chaque round, même pour la référence. Pas de dropout dans ce benchmark. Mesure RAM prudente pour K=20 ; surveiller la mémoire du PC. Si insuffisante, interrompre et conserver le fichier partiel.

La robustesse sur données réelles peut être testée séparément avec l'option V7B `--dropout` (un client fixe manque l'upload à chaque round, mais le contrôleur réalise toujours le calcul local). Ne pas confondre avec une déconnexion après upload ni avec une panne aléatoire.
