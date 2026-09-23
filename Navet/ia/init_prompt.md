# Mission : agent Snake en Reinforcement Learning (cours STP – EPITA)

## Contexte
Cours « Reinforcement Learning avec Python », librement inspiré de Patrick Loeber
(github.com/patrickloeber/snake-ai-pytorch). Architecture en 3 blocs :
- **Game** (PyGame) : `play_step(action) -> reward, game_over, score`
- **Model** (PyTorch) : réseau Q (Linear_QNet / DQN), `predict(state) -> action`
- **Agent** : boucle get_state → get_move → play_step → get_state → remember → train

Cette semaine : version RL. Semaine prochaine : version algorithmique (Dijkstra / Greedy
Best-First Search), puis une version maison « meilleure que les deux combinées ».
L'évaluation porte sur : meilleur score, jeu terminé ou non (grille remplie), **temps pour y
arriver**, courbe d'apprentissage, et un changelog/timeline du travail.

## Contraintes NON négociables (équilibre entre groupes)
1. Ne pas modifier la clock (vitesse/tick du jeu).
2. Ne pas modifier les dimensions de la grille — y compris pour l'entraînement (pas de
   curriculum sur petite grille).
3. Ne pas modifier le scoring du jeu (le score affiché/évalué).
Distinction importante : le **score du jeu** est intouchable ; la **reward interne de
l'agent** est un choix d'implémentation. Toute reward shaping doit être isolée dans l'agent
(ou un wrapper), configurable, documentée, et ne jamais altérer le score affiché. Si un
doute existe sur ce qui est autorisé (reward, mode headless sans tick…), STOP : liste la
question pour que je la valide auprès du prof, et laisse l'option désactivée par défaut.

Barème de référence du cours : pomme +10, défaite −10, fin de jeu +100, autres 0,
« se déplacer +0,1 ».

## Étape 0 — Lire avant d'écrire
Explore d'abord le code existant (structure, boucle de jeu, clock, taille de grille, calcul du
score, gestion de fin de partie). Ne présuppose rien. Produis un court plan avant de coder.

## Problème central à anticiper : la stratégie « en S »
Un agent qui optimise la survie converge vers un parcours en serpentin / cycle hamiltonien :
quasi imbattable mais très lent, donc pénalisé sur le critère temps. Deux causes à traiter :
- **Une reward positive par déplacement (+0,1) récompense le fait de tourner en rond.**
  Ne pas l'utiliser telle quelle : un agent maximise alors la durée, pas la vitesse.
- La pénalité de mort (−10) écrase tout le reste une fois le serpent long → l'agent devient
  ultra-conservateur.

Leviers à implémenter (configurables, testés en ablation) :
- **Coût par pas léger et négatif** (ex. −0,01) à la place du +0,1.
- **Reward shaping basé sur un potentiel** : F = γ·Φ(s') − Φ(s) avec Φ = −distance BFS
  (pas Manhattan) tête→pomme. Préserve la politique optimale, pousse vers la pomme.
- **Compteur de faim** (pas depuis la dernière pomme, normalisé par la surface libre) dans
  l'état, avec troncature d'épisode si trop long — en distinguant bien *truncation* et
  *terminal* dans la cible Q (pas de bootstrap coupé à tort).
- **Tolérance adaptative à la longueur** : plus le serpent est long, moins on pénalise
  l'éloignement de la pomme (il a besoin de manœuvrer), mais le coût du temps reste.
- Métrique dédiée : **pas par pomme** (moyenne et p95) à suivre en continu. Si elle monte
  vers ~la surface de la grille, l'agent dérive vers le S → signaler.

## Erreurs du tuto Loeber à NE PAS reproduire
- **État à 11 booléens aveugle au corps** (danger uniquement sur la case adjacente) → le
  serpent s'enferme dans ses propres boucles. Enrichir l'état.
- **Exploration qui meurt** : ε = 80 − n_games → plus aucune exploration après 80 parties.
  Utiliser une décroissance progressive avec plancher (ou NoisyNets).
- **Pas de target network** → instabilité. Ajouter target network (hard ou soft update).
- **γ = 0,9** : horizon trop court pour planifier sur un long serpent. Tester 0,95–0,99.
- **Timeout 100×len(snake) traité comme une mort à −10** → confond famine et collision.
- **Placement de la pomme par tirage aléatoire récursif** → boucle/crash quand la grille est
  presque pleine. Choisir parmi les cases libres ; gérer proprement la victoire (+100).
- Modèle sauvegardé uniquement au record → garder aussi des checkpoints réguliers.

## État enrichi (proposition, à mesurer)
Garder la version 11 bits comme baseline, puis ajouter :
- Pour chaque action (tout droit / droite / gauche) : **surface accessible par flood fill**
  normalisée, **queue atteignable** (booléen), **distance BFS à la pomme** après l'action.
- Longueur normalisée, compteur de faim normalisé.
Alternative à tester en parallèle : **CNN sur la grille** en canaux (tête, corps avec gradient
d'âge des segments = info « où la queue va se libérer », pomme), orientée **égocentrique**
(rotation selon la direction de la tête).

## Algorithme
Baseline : DQN façon Loeber (pour la courbe « avant »). Puis, par itérations mesurées :
Double DQN → Dueling → Prioritized Replay → n-step returns. En option : PPO avec
**action masking** des coups immédiatement mortels.
Option « bouclier de sécurité » (refuser un coup qui rend la queue inatteignable) : à
implémenter derrière un flag, et reporter les résultats **avec et sans**, pour que la part
« pur RL » reste identifiable — c'est aussi le pont vers la version hybride de la semaine
prochaine.

## Mesures et reproductibilité
- Seeds fixées ; config par run (YAML/JSON) ; un commit git par expérience.
- Log CSV par partie : n_game, score, steps, pas/pomme, temps réel, % grille remplie,
  victoire, cause de mort (mur / corps / famine), ε, loss moyenne.
- Courbes : score, moyenne glissante, pas/pomme, taux de victoire, temps jusqu'à victoire.
- Évaluation séparée de l'entraînement (ε = 0, N parties, clock intacte).

## Livrables
1. Repo GitHub propre (README : lancer l'entraînement, l'évaluation, les courbes).
2. `CHANGELOG.md` horodaté : quand, quoi, pourquoi, résultat (chiffres).
3. Dossier `results/` : CSV + graphes par run, tableau comparatif des ablations.
4. `NOTES_TOUR_DE_TABLE.md` en 3 sections :
   - Difficultés techniques & débogage
   - Obtention et validation des résultats (métriques)
   - Transfert en entreprise
   Alimente-le au fil de l'eau (bugs rencontrés, décisions, surprises).

## Façon de travailler
Itératif : baseline qui tourne → mesure → un changement à la fois → mesure → changelog.
Signale explicitement toute action qui pourrait toucher aux 3 contraintes.