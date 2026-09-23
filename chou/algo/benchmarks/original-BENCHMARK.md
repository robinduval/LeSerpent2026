# Comparaison reproductible des agents Snake

> Rapport historique de la comparaison initiale `fixed` / `explored64`. Voir [BENCHMARK.md](../BENCHMARK.md) pour l’agent actuellement livré. Les données originales sont conservées dans [original-baseline-results.json](original-baseline-results.json).

Mesures sur les graines **1 à 20**, Python 3.13.15.
Même moteur partagé avec le jeu affiché : tore 15×15, corps initial de trois cases, croissance différée et score maximal 223.

## Résultats

| Méthode | Scores | Complètes | Collisions | Interruptions | Déplacements moyens jusqu’à 223 | Durée moyenne équivalente x1 |
|---|---|---:|---:|---:|---:|---:|
| Cycle fixe | 223 pour chaque graine | 20/20 | 0 | 0 | 12498.00 | 2499.60 s |
| Cycle reconfigurable + 2-opt + exploration 64 | 223 pour chaque graine | 20/20 | 0 | 0 | 6016.85 | 1203.37 s |

**Réduction moyenne : 51.86 %** des déplacements. Calcul : 1 − (somme des pas reconfigurés / somme des pas fixes).
Le résultat est proche de la référence annoncée de 52 %, sans démontrer une optimalité globale.

**Meilleure partie reconfigurable : graine 15, score 223, 5700 déplacements, temps x1 1140.0 secondes (19 minutes).**

Les durées x1 sont calculées par `déplacements / 5`. Les 40 parties ont été simulées sans attente ni affichage : ce ne sont pas des chronomètres officiels mesurés en jouant à 5 Hz.
Temps mural total des 40 simulations, contrôles compris : 72.944 secondes.
Le chronomètre de l’interface reste mural ; son premier mouvement précède le premier `clock.tick(5)`. La durée x1 normalisée ne prétend pas reproduire cet écart initial ni les variations du système.

## Coût par décision

| Méthode | Décisions | Moyenne | p95 | Maximum |
|---|---:|---:|---:|---:|
| Cycle fixe | 249960 | 0.095952 ms | 0.121834 ms | 0.818166 ms |
| Reconfigurable 64 | 120337 | 0.208774 ms | 0.157666 ms | 9.417333 ms |

Le coût inclut `select_action`, la recherche, la préparation et les validations internes complètes du cycle. Il exclut la construction du snapshot, le moteur, le rendu et les vérifications externes du benchmark. Le cycle fixe conserve les mêmes vérifications de sécurité.
Le p95 est le percentile empirique au rang supérieur. Les recherches après consommation représentent moins de 5 % des décisions ; leur coût peut porter la moyenne au-dessus du p95.
Le maximum observé de l’agent reste inférieur aux 200 ms disponibles par mouvement à 5 Hz. Cela décrit cette mesure locale, sans garantie de latence sur toute machine.

## Résultats par graine

| Graine | Score fixe | Pas fixes | x1 fixe (s) | Score reconfiguré | Pas reconfigurés | x1 reconfiguré (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 223 | 12849 | 2569.8 | 223 | 5980 | 1196.0 |
| 2 | 223 | 11580 | 2316.0 | 223 | 5764 | 1152.8 |
| 3 | 223 | 12317 | 2463.4 | 223 | 5854 | 1170.8 |
| 4 | 223 | 12896 | 2579.2 | 223 | 6315 | 1263.0 |
| 5 | 223 | 12526 | 2505.2 | 223 | 6519 | 1303.8 |
| 6 | 223 | 12650 | 2530.0 | 223 | 5793 | 1158.6 |
| 7 | 223 | 11952 | 2390.4 | 223 | 6457 | 1291.4 |
| 8 | 223 | 12201 | 2440.2 | 223 | 5857 | 1171.4 |
| 9 | 223 | 12960 | 2592.0 | 223 | 5952 | 1190.4 |
| 10 | 223 | 13186 | 2637.2 | 223 | 6106 | 1221.2 |
| 11 | 223 | 11409 | 2281.8 | 223 | 5795 | 1159.0 |
| 12 | 223 | 12736 | 2547.2 | 223 | 6303 | 1260.6 |
| 13 | 223 | 12524 | 2504.8 | 223 | 5914 | 1182.8 |
| 14 | 223 | 12882 | 2576.4 | 223 | 6342 | 1268.4 |
| 15 | 223 | 13354 | 2670.8 | 223 | 5700 | 1140.0 |
| 16 | 223 | 11579 | 2315.8 | 223 | 5735 | 1147.0 |
| 17 | 223 | 12213 | 2442.6 | 223 | 5900 | 1180.0 |
| 18 | 223 | 11934 | 2386.8 | 223 | 6296 | 1259.2 |
| 19 | 223 | 13515 | 2703.0 | 223 | 5934 | 1186.8 |
| 20 | 223 | 12697 | 2539.4 | 223 | 5821 | 1164.2 |

Les 40 parties sont complètes, sans collision ni interruption. Les mêmes graines ne donnent pas forcément les mêmes positions de pommes entre politiques, car les cases occupées diffèrent. La liste des cases libres et son ordre x-major sont identiques à ceux du moteur.

## Validation et limites

- 27 tests validés : transitions différées, raccordements des quatre bords, unicité du cycle, alignement du corps, conservation du cycle lors des rejets, progression, budget 64, redémarrage, chronomètre et comptabilité des interruptions.
- Séquence adversariale avec l’agent : 223 pommes consécutives, victoire en 223 mouvements, sans collision.
- Invariants contrôlés avant et après chaque mouvement du benchmark. Chaque reconfiguration est aussi validée avant son application en production.
- Attente maximale d’une pomme observée sur les 40 parties : 221 mouvements (borne conservatrice 224).
- Le générateur du moteur n’est pas transmis à l’agent. Celui-ci reçoit seulement un snapshot public immuable et utilise un générateur d’exploration indépendant, déterministe depuis cet état.
- 20 images ont aussi été rendues avec le vrai Pygame, en pilote vidéo sans fenêtre et sans attente ; grille, polices, pomme, serpent et fermeture vérifiés. La capture temporaire n’est plus conservée.
- La garantie de progression dépend du cycle complet compatible avec le corps, pas du seul ordre des segments. Aucun réseau de neurones ni apprentissage.

## Reproduire

```sh
.venv/bin/python -m unittest discover -v
.venv/bin/python serpent-algo.py --benchmark --policies fixed,explored64 --seeds 1:21 --output benchmarks/recheck-original.json
.venv/bin/python serpent-algo.py --policy explored64 --seed 15
```

Données détaillées et protocole : [original-baseline-results.json](original-baseline-results.json).
SHA-256 de `serpent-algo.py` mesuré : `67bb0baebdfd4d3bee328a7551ff1884df6ddc7d56c4ab65660b2e2636a38a55`.
