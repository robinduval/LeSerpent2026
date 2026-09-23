# Résultats et validation de l’agent Snake

**Agent livré : `bridge_uphill`. Moyenne réservée : 3 626,32 déplacements, soit 725,26 secondes x1.** La cible de 4 500 déplacements est atteinte ; celle de **600 secondes / 3 000 déplacements est non atteinte**. Le gain sur le comparateur précédent `all128`, rejoué sur les mêmes graines, est de **25,55 %**. Aucun optimum global n’est revendiqué.

## Meilleurs temps enregistrés

**Record toutes variantes : 612,2 secondes x1 — 10 min 12,2 s — à 223 points.**

| Périmètre | Politique | Graine | Score | Déplacements | Temps équivalent x1 |
|---|---|---:|---:|---:|---:|
| Meilleure partie parmi les variantes mesurées | `bridge128` | **2081** | **223** | **3 061** | **612,2 s** |
| Meilleure partie de l’agent livré par défaut | `bridge_uphill` | **2010** | **223** | **3 152** | **630,4 s** |

Ces deux parties sont complètes, sans collision ni interruption. Source : [résultats des 100 graines réservées](benchmarks/sprint10/heldout.json). Les temps sont calculés par `déplacements / 5` depuis des simulations accélérées ; ce ne sont pas des chronométrages réels de l’interface.

`bridge_uphill` reste le choix par défaut parce que sa moyenne est meilleure : **725,26 s**, contre **729,86 s** pour `bridge128`. Le record individuel ne remplace pas la comparaison des moyennes. L’échéance de recherche peut modifier le parcours lors d’une relance avec la même graine ; le temps exact n’est donc pas garanti.

## Changement principal

Le voisinage comprend maintenant une transformation **4-opt** : `A+B+C+D+E → A+D+C+B+E`, uniquement sur l’arc libre. Les quatre nouvelles connexions toriques sont vérifiées. Les blocs gardent leur orientation, toutes les cases restent présentes exactement une fois et le corps reste contigu. Les améliorations 2-opt et Or-opt sont conservées.

La recherche autorise des détours **virtuels** jusqu’à huit rangs au-delà du meilleur cycle, puis reprend les descentes locales. Seul le meilleur cycle est finalement adopté ; les déplacements réels restent strictement progressifs. Quatre branches partagent 256 propositions.

La continuation sûre initiale reste disponible pendant toute recherche. Les cycles candidats et le futur corps sont certifiés avant exécution. La croissance différée, les pommes aléatoires, l’interface, le chronomètre et la cadence **5 mouvements/s** sont inchangés. Aucun apprentissage, aucune pomme future et aucun accès au générateur aléatoire du moteur.

## Protocole et sélection

- Réglages rapides sur **201–203** uniquement : [ablations](benchmarks/sprint10/ABLATIONS.md). Les finalistes et paramètres ont été figés avant le lot réservé.
- Validation indépendante sur **2001–2100** : trois politiques, **300 parties**, six processus concurrents, échéance coopérative 20 ms. Aucun nouveau réglage après observation de ce lot.
- Régression historique **1–20** : les deux finalistes, 40 parties.
- Sélection : complétions et absence d’échecs d’abord, moyenne de déplacements ensuite.
- Tous les mouvements passent par le même `Game.step()` ; invariant du cycle, corps et progression vérifiés pendant les simulations.

Le [protocole](benchmarks/sprint10/protocol.json), la [source figée](benchmarks/sprint10/frozen-agent.py) et la [source précédente](benchmarks/sprint10/baseline-agent.py) sont conservés. Le rapport antérieur reste disponible dans [previous-BENCHMARK.md](benchmarks/sprint10/previous-BENCHMARK.md).

## Cent graines réservées

| Agent | Parties à 223 | Collisions | Interruptions | Moyenne pas | Médiane | p95 | Maximum | Moyenne x1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all128` | 100/100 | 0 | 0 | 4 870,51 | 4 862,5 | 5285 | 5587 | 974,10 s |
| `bridge128` | 100/100 | 0 | 0 | 3 649,30 | 3 671,0 | 3943 | 4180 | 729,86 s |
| `bridge_uphill` | 100/100 | 0 | 0 | 3 626,32 | 3 611,5 | 3960 | 4191 | 725,26 s |

Parties réservées les plus lentes du gagnant :

| Graine | Déplacements | Durée équivalente x1 |
|---:|---:|---:|
| 2030 | 4191 | 838,2 s |
| 2077 | 4072 | 814,4 s |
| 2073 | 4065 | 813,0 s |
| 2053 | 4019 | 803,8 s |
| 2098 | 3997 | 799,4 s |

Résultats complets : [heldout.json](benchmarks/sprint10/heldout.json). Le champ générique `movement_reduction_percent` du JSON compare première et dernière politique ; le gain du gagnant ci-dessus est recalculé explicitement depuis ses moyennes.

## Régression 1–20

| Agent | Parties à 223 | Collisions | Interruptions | Moyenne pas | Médiane | p95 | Maximum | Moyenne x1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bridge128` | 20/20 | 0 | 0 | 3 698,25 | 3 675,5 | 4032 | 4185 | 739,65 s |
| `bridge_uphill` | 20/20 | 0 | 0 | 3 641,40 | 3 663,0 | 3915 | 3926 | 728,28 s |

Données : [regression.json](benchmarks/sprint10/regression.json).

## Coût des décisions sans concurrence

Mesure complémentaire sur les graines **2001–2002**, après la campagne, avec un seul processus. Les deux parties atteignent 223 points sans échec.

| Décision de `bridge_uphill` | Nombre | Moyenne ms | p95 ms | Maximum ms |
|---|---:|---:|---:|---:|
| Ordinaire | 6 142 | 0,437 | 0,916 | 2,071 |
| Avec recherche | 446 | 18,306 | 20,660 | 21,455 |

Données : [latency-serial.json](benchmarks/sprint10/latency-serial.json). Ce petit lot mesure le coût des décisions ; il ne remplace pas les 100 graines de comparaison des déplacements.

## Coût des décisions sous charge concurrente

Ces latences sont mesurées pendant les **six processus concurrents** du lot réservé. Elles ne sont pas présentées comme des latences isolées du jeu interactif.

| Agent / décision | Nombre | Moyenne ms | p95 ms | Maximum ms |
|---|---:|---:|---:|---:|
| `all128` / ordinary | 464751 | 0,301 | 0,389 | 6,111 |
| `all128` / search | 22300 | 13,954 | 20,413 | 22,171 |
| `bridge_uphill` / ordinary | 340332 | 0,513 | 1,076 | 15,621 |
| `bridge_uphill` / search | 22300 | 18,566 | 20,838 | 22,331 |

Le coût inclut la décision et ses validations internes, mais exclut le rendu, le moteur et les contrôles externes du benchmark. Les recherches coûteuses sont séparées des déplacements ordinaires. L’échéance de 20 ms est coopérative : une énumération ou une validation en cours et la préparation finale peuvent finir au-delà. Son expiration conserve le meilleur parcours certifié et ne compte pas comme interruption de partie.

## Temps et reproductibilité

Les durées x1 sont **calculées par déplacements / 5**, à partir de simulations accélérées, et ne sont pas des chronométrages réels de l’interface. Les campagnes ont duré 278,93 secondes murales pour le lot réservé et 121,00 secondes pour la régression, contrôles compris.

Les graines des pommes et le hasard exploratoire sont indépendants. Une échéance murale rend le parcours sensible à la charge du système ; une même graine peut donc produire un nombre de mouvements différent lors d’une relance. `--time-limit-ms 0` conserve seulement le plafond de travail reproductible, sans modifier la cadence du jeu.

Les **53 tests passent en 13,187 s** et couvrent le moteur, la croissance différée, les bords, la victoire, le redémarrage, les invariants et les transformations. Les six nouveaux tests vérifient notamment les ponts 4-opt, leurs rangs, le budget et l’expiration sûre. Le moteur et le rendu existants n’ont pas été modifiés pendant ce sprint. Le vrai Pygame a également été contrôlé sur dix images pour chacun des deux finalistes, cadence demandée 5 Hz et fermeture correcte : [ui.json](benchmarks/sprint10/ui.json).

## Commandes

```sh
.venv/bin/python serpent-algo.py
.venv/bin/python serpent-algo.py --seed 2010
.venv/bin/python serpent-algo.py --policy bridge128 --seed 2081
.venv/bin/python serpent-algo.py --policy all128
.venv/bin/python -m unittest discover -v
.venv/bin/python serpent-algo.py --benchmark --policies all128,bridge128,bridge_uphill --seeds 2001:2101 --workers 6 --time-limit-ms 20 --output benchmarks/sprint10/recheck-heldout.json
```

## Organisation des résultats conservés

- [benchmarks/sprint10](benchmarks/sprint10) contient la campagne actuelle : protocole, résultats réservés et de régression, sélection, latences et vérification du rendu.
- [benchmarks/sprint10/tuning](benchmarks/sprint10/tuning) regroupe les mesures de réglage et les variantes écartées. Les scripts de prototype temporaires ont été supprimés ; les résultats bruts sont conservés.
- Les sources figées de mesure restent archivées pour la traçabilité. Elles ne sont pas utilisées par l’agent lancé depuis `serpent-algo.py`.
- [benchmark-results.json](benchmark-results.json) correspond à la **campagne précédente**, sur les graines 1001–1100. Ses chiffres et les anciens rapports restent des références historiques ; la campagne actuelle utilise 2001–2100.
