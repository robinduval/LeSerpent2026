# Ablations de réglage de l’agent Snake

Ce document utilise exclusivement les six fichiers `tuning-*.json` terminés. Il ne contient aucun résultat des graines de régression 1–20 ni des graines réservées 1001–1100. Les conclusions ci-dessous servent à retenir deux finalistes ; elles ne désignent pas le gagnant de la validation réservée.

Les **125 parties de réglage enregistrées atteignent toutes 223 points**, sans collision ni interruption. Il s’agit de plusieurs évaluations sur les mêmes dix graines de réglage, et non de 125 graines indépendantes. Les règles restent celles du moteur commun : tore 15×15, croissance différée et 5 mouvements/s. Toutes les simulations sont accélérées ; la durée équivalente x1 vaut `déplacements / 5`, sans prétendre être un chronométrage du jeu affiché.

Les comparaisons emploient les mêmes graines à l’intérieur de chaque tableau. Deux politiques peuvent néanmoins rencontrer des pommes différentes, car leurs corps occupent des cases différentes. Le budget de 64 désigne un plafond de propositions exploratoires, auquel s’ajoutent les descentes locales et les validations : il ne correspond pas à un temps de calcul identique.

## 1. Comparaison initiale à budget 64, sans limite murale

Graines **101–105**. Les cinq lignes nouvelles utilisent quatre branches et un ensemble de cycles normalisés déjà visités. La référence est la marche exploratoire historique de 64 transformations. Ses chiffres sont recalculés sur les cinq mêmes graines depuis le fichier de référence de dix graines.

| Variante | Déplacements moyens | Médiane | Maximum | Décision moyenne (ms) | Recherche moyenne (ms) |
|---|---:|---:|---:|---:|---:|
| `explored64` : référence historique | 6 002,8 | 5 969 | 6 398 | 0,203 | 2,333 |
| `beam64` : quatre branches et suppression des revisites | 6 035,4 | 6 031 | 6 539 | 0,277 | 3,341 |
| `neutral64` : ajout des inversions neutres contenant la pomme | 5 938,2 | 6 031 | 6 161 | 0,280 | 3,357 |
| `oropt64` : ajout des déplacements de petits blocs | 4 985,0 | 5 017 | 5 305 | 0,659 | 10,362 |
| `adaptive64` : ajout de recherches en cours de trajet | 4 835,8 | 4 751 | 5 237 | 1,077 | 10,614 |
| `lookahead64` : ajout d’une anticipation de six mouvements | 4 926,8 | 4 929 | 5 237 | 2,713 | 2,713¹ |

¹ L’anticipation est exécutée à chaque décision : toutes les décisions de cette variante sont classées comme recherches. Cette moyenne ne représente donc pas uniquement les explorations à l’apparition des pommes.

L’effet des branches seules est défavorable sur ce petit lot : **+0,54 %** de mouvements par rapport à la référence. Les neutres contenant la pomme améliorent ensuite ce résultat de **1,61 %**. Le principal gain est apporté par les petits blocs Or-opt : **−16,05 %** par rapport à `neutral64`.

L’adaptation pendant le trajet gagne **2,99 %** supplémentaires, mais fait passer le nombre de décisions avec recherche de 1 115 à 2 027. L’anticipation ajoutée à cette variante augmente les déplacements de **1,88 %** et multiplie le coût moyen des décisions par environ 2,5 ; elle n’est pas retenue sur cette base.

Sources : [référence](tuning-baseline.json), [ablation initiale](tuning-ablations.json).

## 2. Tous les blocs Or-opt, sans limite murale

Graines **101–105**. Les blocs peuvent avoir toute longueur admissible dans l’arc libre ; leurs raccordements sont vérifiés. Les inversions 2-opt et les quatre branches sont conservées.

| Variante | Budget | Déplacements moyens | Médiane | Maximum | Décision moyenne (ms) | Recherche moyenne / p95 / max (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `all64` | 64 | 4 845,4 | 4 879 | 4 933 | 1,066 | 17,800 / 39,024 / 61,452 |
| `alladaptive64` | 64 | 4 720,2 | 4 677 | 5 032 | 2,221 | 17,937 / 34,181 / 61,578 |
| `all128` | 128 | 4 624,8 | 4 601 | 4 873 | 1,894 | 34,183 / 74,023 / 94,275 |
| `alllookahead64` : anticipation à racines multiples | 64 | 4 799,6 | 4 896 | 5 200 | 3,226 | 3,226 / 3,747 / 66,614¹ |

L’extension à toutes les longueurs améliore `oropt64` de **2,80 %** à plafond exploratoire identique. L’adaptation gagne **2,58 %** par rapport à `all64`, avec 2 620 recherches contre 1 115. Doubler le plafond de 64 à 128 donne **−4,55 %**, mais le maximum mesuré d’une recherche atteint 94,275 ms.

L’anticipation à racines multiples gagne seulement **0,95 %** contre `all64`, avec un coût moyen par décision environ trois fois supérieur et une partie maximale plus lente. Comme dans le tableau précédent, toutes ses décisions sont classées en recherche. Ce compromis n’a pas été retenu parmi les finalistes.

Source : [toutes les longueurs Or-opt](tuning-allblocks.json).

## 3. Recherche avec garde murale de 20 ms

Graines **101–105**. Le plafond de propositions reste actif, mais l’échéance murale peut interrompre l’exploration avant de l’atteindre. `three64` ajoute une reconnexion de trois arêtes, réalisée par deux inversions de blocs libres adjacents.

| Variante | Budget | Déplacements moyens | Médiane | Maximum | Décision moyenne (ms) | Recherche moyenne (ms) | Décisions ayant atteint l’échéance |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all128` | 128 | 4 732,6 | 4 702 | 5 230 | 0,977 | 15,444 | 724 |
| `three64` | 64 | 4 636,2 | 4 573 | 4 988 | 0,986 | 14,497 | 618 |
| `threeadaptive64` | 64 | 4 891,8 | 4 782 | 5 343 | 2,104 | 16,992 | 1 707 |
| `three128` | 128 | 4 714,8 | 4 786 | 4 973 | 1,057 | 16,170 | 739 |
| `frequent64` : déclenchement adaptatif plus fréquent | 64 | 4 877,2 | 4 901 | 5 021 | 3,657 | 17,435 | 3 057 |

Le `all128` limité à 20 ms fait **2,33 %** de mouvements supplémentaires par rapport à son évaluation sans limite murale sur ces graines. Le temps moyen de recherche baisse de 34,183 à 15,444 ms. Cette comparaison décrit les mesures de deux exécutions ; elle ne garantit pas de retrouver la même trajectoire avec une échéance réelle sur une autre machine.

`three64` constitue un finaliste intéressant face à `all128`, mais cette comparaison modifie à la fois le voisinage et le plafond de propositions. Elle ne permet pas d’attribuer isolément toute la différence au 3-opt. L’adaptation plus fréquente et l’augmentation à 128 propositions de la variante 3-opt n’apportent ici aucun gain de déplacements. Elles sont écartées de la sélection finale.

Source : [garde de 20 ms](tuning-deadline20.json).

## 4. Budgets plus élevés avec garde de 40 ms

Graines **101–105**. Ces essais modifient simultanément le plafond de travail et la garde murale ; `all512` utilise aussi huit branches. Ils ne constituent donc pas une mesure isolée du passage de 20 à 40 ms.

| Variante | Budget / branches | Déplacements moyens | Médiane | Maximum | Recherche moyenne / max (ms) | Décisions ayant atteint l’échéance |
|---|---:|---:|---:|---:|---:|---:|
| `all256` | 256 / 4 | 4 836,2 | 4 769 | 5 108 | 25,823 / 40,500 | 669 |
| `all512` | 512 / 8 | 4 930,8 | 4 914 | 5 240 | 26,075 / 40,470 | 673 |
| `three256` | 256 / 4 | 5 016,4 | 4 993 | 5 311 | 25,175 / 40,551 | 655 |

Ces variantes dépensent davantage de temps sans améliorer les meilleures moyennes déjà mesurées. Elles ne sont pas retenues. Une recherche plus longue peut choisir un autre cycle, puis produire un autre parcours et d’autres pommes ; les résultats d’une partie complète ne sont donc pas monotones avec le budget.

Source : [garde de 40 ms](tuning-budget40.json).

## 5. Confirmation sur dix graines de réglage

Graines **101–110**, garde de **20 ms**, exécution sérielle. La référence historique conserve son plafond de 64 et son fonctionnement antérieur. `diverse64` essaie de conserver une configuration neutre distincte lorsqu’une descente rejoint un optimum déjà visité.

| Variante | Déplacements moyens | Médiane | Maximum | Durée moyenne x1 (s) | Gain de déplacements contre la référence |
|---|---:|---:|---:|---:|---:|
| `explored64` | 6 021,9 | 5 958,0 | 6 398 | 1 204,38 | — |
| `all128` | 4 767,5 | 4 779,0 | 5 230 | 953,50 | 20,83 % |
| `three64` | 4 777,5 | 4 725,0 | 5 202 | 955,50 | 20,66 % |
| `diverse64` | 4 851,3 | 4 846,5 | 5 170 | 970,26 | 19,44 % |

| Variante | Décision moyenne (ms) | Recherche moyenne / p95 / max (ms) | Mouvement ordinaire moyen / p95 / max (ms) | Échéances atteintes |
|---|---:|---:|---:|---:|
| `explored64` | 0,200 | 2,283 / 6,813 / 8,246 | 0,120 / 0,140 / 0,355 | 0 |
| `all128` | 0,898 | 13,908 / 20,349 / 20,463 | 0,259 / 0,333 / 4,052 | 1 430 |
| `three64` | 0,934 | 13,848 / 20,399 / 20,550 | 0,302 / 0,411 / 4,742 | 1 235 |
| `diverse64` | 0,917 | 13,668 / 20,407 / 20,532 | 0,302 / 0,411 / 4,861 | 1 259 |

Les deux finalistes retenus pour la validation réservée sont **`all128` et `three64`**. Leurs dix mouvements d’écart moyen sur ce lot sont insuffisants pour annoncer un vainqueur stable. `diverse64` est écarté : sa moyenne est plus élevée malgré un maximum légèrement inférieur. Les branches multiples, les empreintes, les inversions neutres contenant la pomme et Or-opt toutes longueurs sont communs aux deux finalistes. Aucun réglage ne doit être effectué à partir du lot réservé.

Le milieu de partie reste prépondérant : entre les pommes 51 et 150, les moyennes sont de 3 475,9 mouvements pour la référence, 2 671,1 pour `all128` et 2 713,0 pour `three64`, soit encore environ 56–57 % du trajet des finalistes. La relance adaptative a été expérimentée sur ce segment ; son coût et ses résultats sous garde de 20 ms ne justifient pas sa conservation dans les deux configurations finalistes.

**Aucune moyenne de réglage présentée ici n’atteint la cible de 4 500 mouvements.** La meilleure moyenne sans garde murale vaut 4 624,8 sur cinq graines ; la meilleure moyenne des deux finalistes sous garde de 20 ms vaut 4 767,5 sur dix graines. Cela ne préjuge pas des résultats réservés et ne constitue aucune preuve d’optimalité globale.

Sources : [confirmation des finalistes](tuning-finalists.json), [référence de dix graines](tuning-baseline.json).

## Lecture des latences et limites de ces comparaisons

- Les recherches après consommation représentent moins de 5 % des mouvements de plusieurs variantes. Leur coût peut donc être masqué par le p95 global ; les tableaux distinguent explicitement les recherches des mouvements ordinaires.
- Les p95 sont les percentiles empiriques au rang supérieur. Avec cinq ou dix parties, le p95 des déplacements coïncide avec le maximum ; ce petit échantillon ne mesure pas finement les événements rares.
- Une échéance de 20 ou 40 ms est une garde coopérative de recherche. L’énumération et la certification en cours, puis la validation finale du mouvement, peuvent se terminer après son expiration. Les maxima observés légèrement supérieurs à la garde sont donc rapportés tels quels.
- Le plafond de propositions est reproductible. Lorsqu’une garde murale expire, l’ordonnancement du système peut modifier le nombre de propositions réellement explorées, puis les mouvements de la partie. Les parcours sans limite murale et avec limite sont distingués ; il serait incorrect de promettre la reproduction exacte d’une partie limitée par son seul numéro de graine.
- Les fichiers antérieurs n’enregistrent pas tous le champ `workers`; seul le fichier final enregistre explicitement `workers: 1`. Les latences présentées décrivent ces exécutions locales, sans garantie sur une autre machine. Le fichier final utilise Python 3.13.15 et mesure `select_action`, validations internes comprises, en excluant moteur, rendu et contrôles externes.
- Les six JSON conservent chacun leurs configurations et leur empreinte SHA-256 de source. Les versions évoluent pendant les ablations ; les gains au sein d’une même table sont mieux contrôlés que les différences entre campagnes. Les anciens compteurs `revisits` à zéro et les compteurs de propositions de la référence ne permettent pas, à eux seuls, de conclure à une absence de revisites ou de recherche : cette instrumentation n’était pas présente dans toutes les versions.
