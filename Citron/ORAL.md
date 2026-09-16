# Préparation de l'oral — Snake RL

## Noms à montrer dans le code

- `Game` : bloc Game (PyGame), `play_step(action)` renvoie
  `(reward, game_over, score)` comme dans les slides.
- `Linear_QNet` : bloc Model (Torch), réseau neuronal avec `predict(state)`.
- `Agent` : `model`, `get_state(game)`, `get_move(state)`, `remember(...)` et
  `learn(transition)` pour l'entraînement Double DQN.

La boucle exécute `get_state` → `get_move` → `play_step` → `get_state` → `learn`.
`learn` appelle `remember`, puis effectue une mise à jour sur un batch de mémoire.
`model.train()` dans PyTorch signifie activer le mode entraînement du module ;
cette méthode seule ne fait pas l'optimisation. Celle-ci est dans `Agent.learn`.
`Environment`, `step` et `act` restent des alias/adaptateurs de compatibilité pour
les anciens outils ; la boucle principale utilise les noms des slides.

## Version à dire en trois minutes

« Nous sommes partis du jeu Snake fourni. Nous avons conservé la grille 15 × 15,
les bords traversables, la croissance du serpent et le score d'un point par pomme.
Notre objectif est d'atteindre au moins 10 points et de maximiser le meilleur score
sur une batterie de parties. À score égal, le temps permet de départager.

Nous avons séparé le jeu de l'agent. Le jeu reçoit une action, avance d'une case,
puis renvoie la nouvelle observation, une récompense et l'indication de fin.
L'agent choisit entre avancer, tourner à droite et tourner à gauche. Ces actions
relatives respectent l'interdiction du demi-tour du jeu initial.

L'agent utilise un Double DQN. C'est un petit réseau qui estime, pour chaque
action, la somme des récompenses futures qu'il peut espérer. Il apprend à partir
de ses expériences, sans exemples d'un joueur expert. Pendant l'entraînement,
il joue parfois au hasard pour découvrir de nouvelles situations.

Pour stabiliser l'apprentissage, nous conservons les expériences dans une mémoire
et nous en tirons des groupes aléatoires. Nous utilisons deux réseaux : le réseau
principal choisit l'action suivante, et un réseau cible, mis à jour moins souvent,
évalue cette action. Les récompenses sont distinctes du score : une pomme vaut
un point au jeu, mais donne une récompense de +10 pour apprendre.

Au départ, l'état contenait 13 valeurs décrivant les dangers immédiats, la
direction, la position relative de la pomme, la longueur et la croissance en
attente. L'agent a appris rapidement, puis ses scores ont plafonné autour de
22 points en moyenne à l'entraînement. Notre hypothèse était qu'il manquait
d'informations pour anticiper les zones dans lesquelles il pouvait s'enfermer.

Nous sommes donc passés à 22 valeurs, avec l'espace accessible après chaque
action, des distances au corps et des distances à la pomme. Un flood fill mesure
l'espace disponible ; il ne choisit pas l'action. Le réseau reste le décideur.
Nous avons transféré les anciens poids et initialisé à zéro les poids des nouvelles
entrées, pour conserver le comportement initial avant de poursuivre l'apprentissage.

Après autorisation, nous avons accéléré l'entraînement avec huit environnements
partageant un modèle et une mémoire. Pour la démonstration, le modèle est figé et
le jeu tourne toujours à 5 déplacements par seconde. Nous distinguons le temps
réel d'entraînement des durées équivalentes à 5 Hz affichées pour les simulations.

Le modèle initial a terminé une partie d'évaluation à 19 points. Le modèle spatial
a ensuite atteint 82 points à 5 Hz, mais nous avons interrompu cette partie : nous
ne la présentons donc pas comme un score final. Il reste à terminer une batterie
comparable pour mesurer la fiabilité. »

## Comprendre la boucle d'apprentissage

1. Observer : transformer le plateau en 13 ou 22 nombres.
2. Choisir : action aléatoire avec probabilité epsilon, sinon meilleure estimation Q.
3. Jouer : avancer d'une case dans le jeu inchangé.
4. Mémoriser : (état, action, récompense, état suivant, fin).
5. Apprendre : tirer 64 expériences et corriger les estimations du réseau.
6. Recommencer ; créer une nouvelle partie après une fin naturelle.

La cible pour une transition non terminale est :

    récompense + 0,99 × Q_cible(état_suivant, argmax Q_principal(état_suivant))

Si la partie est terminée, la cible est seulement la récompense immédiate.
0,99 est le facteur d'actualisation : les récompenses futures comptent fortement,
mais moins que la même récompense immédiate. Le réseau ne connaît pas la vraie
valeur future ; il la réestime progressivement à partir des expériences.

## Paramètres effectivement utilisés

| Élément | Choix |
|---|---|
| Réseau | 13 → 128 → 3, puis 22 → 128 → 3 ; activation ReLU |
| Optimiseur | Adam, learning rate 0,0003 |
| Récompenses | déplacement +0,1 ; pomme +10 ; mort -10 ; victoire +100 |
| Mémoire | 20 000 transitions, tirage uniforme |
| Apprentissage | batch 64, une mise à jour par transition après remplissage minimal |
| Exploration | epsilon de 1 à 0,05 sur 8 000 transitions cumulées |
| Réseau cible | copie des poids principaux toutes les 500 transitions |
| Stabilisation | loss de Huber et norme des gradients limitée à 10 |
| Parallélisme | 8 jeux entrelacés sur CPU, inférence groupée, modèle commun |
| Évaluation | epsilon 0, aucun apprentissage, un jeu à 5 Hz |

Les huit jeux ne sont pas huit modèles indépendants et ne sont pas huit processus.
Ils diversifient la collecte d'expériences. Nous n'avons pas mesuré séparément le
gain imputable au multi-environnement et celui imputable à l'accélération de clock.

## Pourquoi 22 entrées ?

Les 13 premières : 3 dangers immédiats + 4 directions + 4 indicateurs de pomme
+ 1 longueur normalisée + 1 indicateur de croissance en attente.
Les 9 nouvelles : 3 proportions d'espace accessible + 3 distances au corps
+ 2 distances signées à la pomme + 1 distance de Manhattan torique.

Les bords sont traversables : les distances et le flood fill doivent le respecter.
Une pomme située à gauche sur l'écran peut être plus proche en traversant le bord
droit. La queue libère sa case sauf si une croissance est en attente : le calcul
du danger et de l'espace tient compte de cette règle.

Le flood fill explore les cases libres connectées à une position, comme une
peinture qui se propage sans traverser le corps. Nous normalisons leur nombre
par 225. C'est une approximation statique : le corps continuera à bouger. Ce
calcul n'est ni une garantie de sécurité, ni un masque interdisant des actions.

## Ce qui a changé, dans l'ordre

- Socle manuel → environnement pilotable + agent Double DQN + journalisation.
- Première partie autonome : 2 points, 108,5 secondes à 5 Hz.
- Autorisation d'accélérer → 8 environnements, durées réelles et équivalentes séparées.
- Première évaluation figée : 19 points en 37 secondes.
- Batch massif baseline : record d'entraînement 61, moyenne autour de 22.
- Diagnostic : visibilité spatiale limitée ; hypothèse, pas causalité démontrée.
- Variante spatiale + cache des observations + checkpoints intermédiaires conservés.
- Test des règles et du transfert : neuf tests réussis.
- Baseline massif figé, seeds 1000–1002 : 25, 13, 16 points.
- Variante spatiale figée, seed 1000 : 82 points atteints puis fermeture manuelle,
  avant une fin naturelle ; batterie spatiale incomplète.
- Nouvel entraînement spatial massif en cours au moment de cette fiche.

La variante conserve les anciens poids sur les 13 entrées et met les neuf
nouvelles colonnes à zéro. Le replay et l'optimiseur sont réinitialisés, car nous
ne pouvons pas reconstruire les nouvelles observations à partir des anciennes
transitions. L'epsilon et les compteurs cumulés sont conservés. Ce redémarrage de
mémoire est un facteur supplémentaire : la comparaison n'est pas une ablation pure.

## Questions probables

**Pourquoi pas uniquement plus d'entraînement ?** Deux plateaux différents peuvent
avoir les mêmes 13 valeurs, mais exiger des décisions différentes. Accumuler des
expériences ne suffit pas à lever une ambiguïté dans ce que l'agent observe.

**Pourquoi Double DQN ?** Il sépare la sélection de l'action et son évaluation
dans la cible d'apprentissage, pour limiter la surestimation des valeurs Q.

**Pourquoi une mémoire ?** Tirer des expériences variées évite d'apprendre
uniquement sur une suite de mouvements très corrélés et permet de réutiliser les
expériences. Cela ne garantit pas la convergence.

**Pourquoi pas une grande image et un CNN ?** Le petit vecteur exploite directement
les informations utiles du jeu, avec un coût de calcul et une mise en place faibles.
Il reste toutefois une description partielle du plateau.

**Avez-vous codé la solution du Snake ?** Non. Nous calculons des observations et
les règles du jeu. Aucune recherche de chemin ne choisit les actions du serpent.

**Le réseau apprend-il pendant la démonstration ?** Non : poids figés, epsilon 0.

**Comment lire score/seconde ?** En entraînement, score / (déplacements / 5) est
un équivalent théorique de la trajectoire. En évaluation, c'est score / secondes
réellement mesurées. Le temps du projet est toujours le temps réel.

**82 points prouve-t-il la supériorité ?** C'est un signe encourageant sur une seed,
mais une partie interrompue et une comparaison trop petite ne prouvent pas une
supériorité générale. Il faut plusieurs parties terminées sur les mêmes seeds.

**Pourquoi le record d'entraînement et le modèle final diffèrent ?** Les poids
changent pendant l'entraînement et des actions aléatoires sont encore possibles.
Le dernier checkpoint n'est pas nécessairement le meilleur.

**Application en entreprise ?** Même démarche expérimentale pour une politique
de décision : définir objectif et contraintes, instrumenter, établir une référence,
tester une hypothèse, puis valider sur des situations séparées de l'entraînement.

## Avant l'oral

Actualiser les résultats du dernier batch. Choisir un checkpoint évalué. Afficher
un résultat terminé ou préciser explicitement une interruption. Garder le même
nombre de parties et les mêmes seeds entre candidats. Ne pas présenter la loss
comme le score du jeu, ni une vitesse simulée comme du temps réel observé.
