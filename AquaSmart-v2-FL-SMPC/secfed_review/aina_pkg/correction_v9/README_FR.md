# SecFed — V9A, tests de sécurité adversariaux

Cette étape **ajoute uniquement** `correction_v9/` au projet. Elle requiert `correction_v7/vendor/fl_smpc` installé par l'archive V7A. **Ne pas modifier V5–V8**, dont les manifestes et résultats sont figés.

## Installation (Windows CMD)

Dans `C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC\secfed_review\aina_pkg` :

```cmd
powershell -NoProfile -Command "Expand-Archive -LiteralPath (Join-Path $env:USERPROFILE 'Downloads\SecFed_correction_etape9A.zip') -DestinationPath (Get-Location).Path -Force"
python -m unittest discover -s correction_v9/tests -v
python correction_v9\run_security_validation.py --repeats 3 --output security_results\v9a_security.json
```

Pour un premier essai : `python correction_v9\run_security_validation.py --repeats 1 --output security_results\v9a_smoke.json`.

Si un résultat existe, choisir un autre `--output` pour ne pas l'écraser. Le script écrit le rapport même en cas de test de sécurité raté, et retourne un code non nul en cas d'échec.

Archiver :

```cmd
powershell -NoProfile -Command "Compress-Archive -Path 'security_results\*.json' -DestinationPath 'SecFed_V9_security_results.zip' -Force"
```

## Les 8 scénarios par répétition

| Test | Attendu | Signification |
| --- | --- | --- |
| `normal_mtls` | Succès | Somme correcte via mTLS TLS 1.3 |
| `dropout_before_upload` | Succès | Correction des masques des clients n'ayant jamais uploadé |
| `forged_signature_direct` | Rejet | Signature Ed25519 invalide |
| `modified_signed_payload_direct` | Rejet | Intégrité du vecteur masqué |
| `stale_announcement_replay` | Rejet | Isolation par identifiant de session |
| `forged_upload_over_mtls` | Rejet | Falsification de signature pendant une session réseau réelle |
| `late_dropout_after_upload` | Abandon sécurisé | Le serveur n'émet pas d'agrégat sans preuve d'un survivant qui avait uploadé |
| `insufficient_survivors` | Rejet | Interdiction de révéler un agrégat avec moins de deux contributeurs |

**Attention** : "réussi" pour les attaques signifie que l'exécution a **rejeté** la demande ; un résultat retourné par l'API serait un échec de sécurité. Les tests ne sont pas des preuves cryptographiques.

## Portée et limites

- Le contrôleur d'expérience connaît toutes les contributions individuelles. Il ne s'agit pas de confidentialité de bout en bout contre ce contrôleur.
- Clients et serveur exécutés en processus séparés sur `127.0.0.1`, avec certificats d'une autorité de test éphémère ; **ce n'est pas un déploiement IoT**.
- Le serveur est supposé honnête dans la présentation des uploads et du roster aux clients. La gestion d'un serveur malveillant et de collusions n'a pas été validée.
- Les identités Ed25519 sont annoncées durant la session et ne sont pas liées à un registre persistant de clés de confiance.
- L'abandon sur dropout tardif n'est pas une récupération à seuil. Seul le dropout avant upload peut être récupéré.
- Aucun modèle n'est entraîné ; seules des mises à jour **synthétiques** servent à vérifier le protocole. Ne pas présenter cette étape comme une validation de précision ML ou comme une preuve formelle de SMPC.
- Les erreurs sont numériques avec quantification au facteur `10^8` ; elles n'évaluent ni la confidentialité informationnelle, ni des attaques d'inférence sur les modèles.

Voir `positionnement_scientifique.md` pour les revendications raisonnables et les références.
