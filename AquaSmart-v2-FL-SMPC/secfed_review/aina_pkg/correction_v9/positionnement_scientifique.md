# SecFed — V9 : positionnement scientifique provisoire

## Originalité à revendiquer (expérimentale, non cryptographique)

1. Protocole d'évaluation temporelle synchronisé et séparant la sélection des configurations du test rétrospectif, sur 8 bassins aquacoles. Les mesures répétées de poids sont regroupées par blocs ; les hypothèses de collecte et la généralisation restent limitées.
2. Transfert de protocole à 15 sites énergétiques BDG2 et évaluation intégrale sur la période de test 2017 ; seule une partie de BDG2 est retenue. Ce n'est pas une indépendance parfaite vis-à-vis des explorations antérieures de l'article.
3. Comparaison de FedAvg, modèles locaux, variantes proximales et personnalisation SGD. Les gains relatifs de personnalisation sont comparés au FedAvg sous les mêmes partitions, sans revendication d'algorithme de personnalisation inédit.
4. Intégration d'une somme masquée à clés éphémères et signatures, avec TCP/mTLS et processus séparés ; fidélité numérique sur 580 rounds normaux + 120 rounds à dropout avant upload ; coûts comparés à une référence mTLS non masquée (25 paires de répétitions).
5. Limites quantitatives documentées : surcoût de communication important (d'environ x9,5 à x36,1 sur le benchmark K=4..20) ; performances aquacoles R² négatives ; le test de sécurité V9A n'est pas une preuve d'une nouvelle construction cryptographique.

## Revendications à EXCLURE

- « Première solution Secure Aggregation » : le principe précède SecFed.
- « Nouvelle méthode de personnalisation » : pas démontré.
- « Preuve de sécurité contre un serveur malveillant / des clients collusifs » : **non**.
- « SMPC générique / partage de secrets à seuil résistant à tout dropout » : **non**.
- « Les données sont invisibles au contrôleur expérimental » : **faux** ; le parent charge et fournit les vecteurs en clair aux processus clients.
- « Confidentialité différentielle garantie de bout en bout » : aucune garantie DP établie pour le pipeline.
- « Réseau IoT et dispositifs physiques validés » : loopback Windows uniquement.
- « Bonnes prédictions aquacoles absolues » : R² de test négatifs.
- « Temps TLS mesuré = surcoût cryptographique pur » : le benchmark incorpore orchestration, gestion des processus, messages et certificats.

## Comparaison aux travaux existants

- Bonawitz et al. 2017 : *Practical Secure Aggregation for Privacy-Preserving Machine Learning*, ACM CCS. Déjà une agrégation sécurisée avec clients déconnectés. https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/
- Bell et al., *Secure Single-Server Aggregation with (Poly)Logarithmic Overhead*. Alternative à l'architecture en graphe complet, utile pour discuter la scalabilité. https://research.google/pubs/secure-single-server-vector-aggregation-with-polylogarithmic-overhead/
- So et al. 2022 : *LightSecAgg: a Lightweight and Versatile Design for Secure Aggregation in Federated Learning*, MLSys. Travaux optimisant notamment la tolérance aux déconnexions. https://proceedings.mlsys.org/paper_files/paper/2022/hash/6c44dc73014d66ba49b28d483a8f8b0d-Abstract.html
- Li et al. 2020 : *Federated Optimization in Heterogeneous Networks* (FedProx), MLSys. https://proceedings.mlsys.org/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html
- Li et al. 2021 : *Ditto: Fair and Robust Federated Learning Through Personalization*, ICML. https://proceedings.mlr.press/v139/li21h.html
- Dinh et al. 2020 : *Personalized Federated Learning with Moreau Envelopes* (pFedMe), NeurIPS. https://proceedings.neurips.cc/paper/2020/hash/f4f1f13c8289ac1b1ee0ff176b56fc60-Abstract.html
- Miller et al. 2020 : *The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition*, Scientific Data 7:368. https://doi.org/10.1038/s41597-020-00712-x

La revue systématique d'articles plus récents, les DOIs et les métadonnées bibliographiques complets, ainsi que la vérification de tous les `bibitem` de la version initiale restent à effectuer avant soumission.

## Angles de rédaction possibles

« A reproducible, resource-aware evaluation of cryptographic secure aggregation and lightweight personalized federated learning across heterogeneous IoT sites ».

L'apport est un **pipeline expérimental et une étude de compromis**. Ne pas attribuer à la combinaison X25519/HKDF/ChaCha20/Ed25519 une nouveauté cryptographique intrinsèque.
