# Sécurité du contrôleur hybride

Le bouclier et ses optimisations sont **programmés**, et ne constituent pas du
reinforcement learning. Le réseau appris choisit seulement parmi leurs actions
admissibles. Cette distinction doit rester visible dans les résultats et dans
la qualification de l'agent.

## Invariant et argument de terminaison

Implémentation : `snake_rl/cycle_shield.py`, classe `RewiredCycleShield`.
L'initialisation est alignée sur le corps officiel : les rangs tête–queue sont
115, 114, 113 sur un cycle hamiltonien des 225 cases du tore 15 × 15.

1. Le cycle contient exactement une fois chaque case et toutes ses arêtes sont
   des déplacements autorisés du moteur. Le corps occupe un segment contigu
   de ce cycle, dans le sens queue → tête.
2. Une transformation 2-opt inverse uniquement un segment libre. Les deux
   nouvelles arêtes sont vérifiées comme adjacentes sur le tore. Les arêtes
   internes du corps et la totalité du cycle sont préservées.
3. `commit(env, action)` réorganise le cycle avant `env.step(action)` : l'action
   choisie devient le successeur de la tête. Ce successeur est libre pour tout
   état non terminal admissible. Une croissance différée ne peut donc pas
   provoquer une collision. À 224 cellules avec croissance en attente, le
   successeur est l'unique case libre et contient la dernière pomme : le moteur
   atteint réellement 223 pommes et 225 cellules.
4. Chaque action admissible diminue strictement la distance de la tête à la
   pomme courante sur le cycle résultant. Les optimisations de l'arc libre ne
   l'augmentent jamais. Ainsi, une pomme est atteinte en un nombre fini de pas,
   indépendamment du choix du réseau parmi les actions admissibles. Une borne
   volontairement large est 224 pas par pomme, soit 49 952 pas pour 223 pommes.

Cet argument suppose une partie démarrant au reset officiel, un bouclier remis
à zéro à chaque reset, et exactement un `commit` avant chaque `env.step`. Il ne
s'applique pas à un corps arbitraire, à un état modifié extérieurement, ni à un
cycle désynchronisé. Il ne prouve pas une durée compétitive ni une première
place. Le réseau ne peut pas sortir de l'ensemble admissible.

`optimize_free_arc` réalise des améliorations 2-opt strictes de la distance à
la pomme. `explore_free_arc` explore aussi des transformations neutres, puis
conserve uniquement une meilleure configuration. Son pseudo-aléatoire local
est dérivé de l'état public ; aucun état RNG du moteur, aucune future pomme et
aucune modification du jeu ne sont consultés ou effectués.

## Variante expérimentale rejetée

L'ancienne fonction `allowed_actions(env)` utilise des rangs fixes et autorise
les raccourcis sans reconstruire un cycle hamiltonien complet. Elle garde le
corps ordonné mais peut laisser des trous dans son arc occupé. Une série de
pommes consécutives et la croissance différée peuvent alors épuiser l'arc libre.
Un diagnostic de 500 parties (250 choix aléatoires, 250 choix gloutons) a donné
17 ensembles d'actions vides. Cette variante reste un comparateur expérimental
explicitement documenté ; **elle ne bénéficie pas de l'argument de sécurité**.
Le contrôleur hybride doit utiliser `RewiredCycleShield`.

## Vérifications et coût de calcul

`tests/test_cycle_shield.py` contient six tests : cycle complet et alignement,
pureté des requêtes et des indices prospectifs, sept parties entières à choix
admissibles aléatoires, séquence adverse de 223 pommes consécutives, statut
explicitement algorithmique du comparateur, et deux parties avec recherche de
64 transformations vérifiant cycle, corps, progression et RNG inchangé.
Tous ont réussi le 16 septembre 2026.

Un diagnostic local de latence du comparateur glouton, graine 15158, a donné :

| Recherche | Pas jusqu'à 223 | p95 par décision | Maximum | Maximum corps > 150 |
|---|---:|---:|---:|---:|
| 64 essais | 6 274 | 0,037 ms | 6,860 ms | 0,083 ms |
| 128 essais | 6 540 | 0,044 ms | 16,343 ms | 0,082 ms |

Ces mesures utilisent `time.perf_counter()` et incluent préparation du cycle,
choix algorithmique et commit, mais pas le réseau ni le rendu. Elles ont été
réalisées en simulation accélérée pour vérifier l'implémentation. **Ce ne sont
pas des évaluations officielles à 5 Hz, ni des scores du réseau entraîné.**
La recherche de 64 essais est retenue comme option peu coûteuse ; 128 n'a pas
amélioré cette fixture.
