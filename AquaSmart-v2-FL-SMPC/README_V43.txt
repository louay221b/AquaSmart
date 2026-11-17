AQUASMART V4.3 — CRYPTOGRAPHIC SECURE AGGREGATION (PROTOTYPE)
============================================================

OBJECTIF
Remplacer l'emulation de masques gaussiens V4.2 par une SOMME SECURISEE
utilisant des primitives cryptographiques standard. La selection du modele
FedAvg et du fine-tuning reste effectuee uniquement sur VALIDATION.

COMPATIBILITE
Ajoute uniquement :
  fl_smpc/secure_aggregation.py
  fl_smpc/run_temporal_experiments_v43.py
  tests/test_temporal_v43.py
  requirements_v43.txt
  README_V43.txt
  SECURITY_MODEL_V43.md
Ne remplace pas les fichiers V2, V3, V4.1, V4.2 existants.

INSTALLATION - WINDOWS CMD
1. Telecharger AquaSmart_V4_3_Crypto_SecAgg.zip dans Downloads.
2. Depuis le dossier AquaSmart-v2-FL-SMPC :

   cd /d C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC
   .venv\Scripts\activate
   powershell -NoProfile -Command "Expand-Archive -LiteralPath (Join-Path $env:USERPROFILE 'Downloads\AquaSmart_V4_3_Crypto_SecAgg.zip') -DestinationPath (Get-Location).Path -Force"
   python -m pip install -r requirements_v43.txt
   python -m py_compile fl_smpc\secure_aggregation.py fl_smpc\run_temporal_experiments_v43.py
   python -m unittest discover -s tests -p "test_temporal_v43.py" -v

3. Premier essai sur les 11 bassins (seed 42) :

   python -m fl_smpc.run_temporal_experiments_v43 --sample-cap 3000 --rounds 5,10,20 --lr-grid 0.005,0.01 --alpha-grid 0.1,1 --epochs-grid 1,5 --personal-lr-grid 0.001,0.005,0.01 --personal-epochs-grid 0,1,3,5 --seeds 42 --selection macro_rmse

4. Seulement apres le bon fonctionnement de la seed 42, 10 seeds :
   (reprendre la commande precedente, et remplacer --seeds 42 par :)
   --seeds 11,12,13,14,15,16,17,18,19,20

SORTIES
fl_smpc/results/temporal_v43/seed42_macro_rmse_<id>.json
fl_smpc/results/temporal_v43/summary_macro_rmse_<id>.json

Methodes principales :
- federated_tuned (FedAvg plain)
- personalized_fedavg (FedAvg + fine-tuning local, plain)
- cryptographic_secagg_fedavg
- cryptographic_secagg_personalized
plus les baselines centralized, local, federated_baseline.

PROTOCOLE REELLEMENT IMPLEMENTE
- Identites de clients Ed25519 (cles de verification pre-enregistrees)
- Cle ephemere X25519 par client et par round, annonce signee
- Secret partage pairwise, HKDF-SHA256, masque ChaCha20 a cle unique
- Encodage fixed-point x10^8 et somme modulo 2^61 - 1
- Mise a jour signee, verification des signatures serveur
- Masques qui s'annulent dans la somme si TOUS les clients participent
- En cas d'absence d'un client, ECHEC FERME (pas de somme partielle)
- Mesure du temps de generation de cles, masquage/signature, verification,
  quantite de messages encodes en memoire (pas de mesure reseau)
- Verifie l'equivalence avec FedAvg/plain personalise (difference finie)

LIMITATIONS DE SECURITE (OBLIGATOIRES POUR L'ARTICLE)
- Tous les clients doivent rester connectes pendant le round.
- Pas de recuperation securisee des abandons, pas de seuil Shamir.
- Pas de resilience aux clients malveillants, collusions, serveur malveillant.
- Pas de confidentialite differentielle, pas de protection contre les
  informations indirectes revelees par la somme globale.
- Benchmark monoprocessus: aucune isolation OS des clients/serveur, aucun
  canal reseau/TLS, aucune gestion reelle de certificats/cles persistantes.
- Mesures de bytes = charges utiles serialisees en memoire + estimation
  envoi du modele. PAS un resultat de bande passante/latence reseau.
- Les cibles feed_ration_g sont synthetiques (proxy, pas du feed mesure).
- Tests temporels deja explores (resultats exploratoires, non finaux).
- Budgets de calcul des baselines non strictement apparies.

PROCHAINE ETAPE
V4.4 : tolerance aux abandons via protocole audite + vraies communications
inter-processus (TLS), menaces et collusions, et benchmarks de latence.
Ensuite, nouvelles fenetres temporelles independantes pour la publication.
