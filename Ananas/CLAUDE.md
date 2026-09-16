# Contexte agent - Projet Le Serpent 2026

## Mission

Construire deux agents autonomes pour le jeu Snake fourni par l'enseignant :

1. `snake-ia.py` : un agent de Deep Reinforcement Learning utilisant réellement un réseau neuronal ;
2. `snake-algo.py` : un agent déterministe de recherche de chemin.

L'objectif produit est de ramasser les pommes rapidement sans collision, puis de finir la grille lorsque cela est possible. L'ordre des priorités est :

1. ne pas mourir ;
2. préserver une trajectoire future ;
3. atteindre la prochaine pomme avec peu de déplacements ;
4. réduire le temps de calcul d'une décision.

Les deux livrables avancent en parallèle. Le moteur du jeu doit rester inchangé ; seules la logique des agents, l'entraînement, les mesures et les sauvegardes de modèles peuvent être ajoutés.

## Sources autorisées pour ce cadrage

- README du dépôt enseignant : <https://github.com/robinduval/LeSerpent2026>
- Support de cours joint : `2026 - stp - cours 2 (ia).pdf`

Ce cadrage n'a pas inspecté le code du dépôt. Avant d'implémenter, lire `serpent.py` pour découvrir ses interfaces sans modifier son comportement.

## Contraintes non négociables

- Ne pas modifier la clock.
- Ne pas modifier les dimensions de la grille.
- Ne pas modifier le scoring du jeu.
- Ne pas remplacer le Deep Learning par un agent à règles dans `snake-ia.py`.
- Ne pas utiliser un cycle hamiltonien comme solution principale.
- Ne pas altérer les règles pour accélérer artificiellement les résultats.
- Conserver le visuel et les mécanismes attendus du jeu de base.
- Isoler toute adaptation derrière une interface d'agent ; ne pas réécrire le moteur sans validation explicite.
- Rendre les expériences reproductibles : seed, configuration, modèle et résultats doivent être enregistrés.

Le reward d'entraînement n'est pas le scoring du jeu. Il peut être expérimenté uniquement si l'enseignant confirme qu'il est ajustable ; le score visible du jeu ne change jamais.

## Architecture cible minimale

Créer une interface conceptuelle commune :

```python
class Agent:
    def reset(self, game): ...
    def choose_action(self, game): ...
    def observe(self, transition): ...
```

La signature exacte doit s'adapter à `serpent.py`. Séparer autant que possible :

- l'adaptateur du jeu ;
- l'encodage de l'état ;
- le choix d'action ;
- l'entraînement DQN ;
- la mesure et la journalisation ;
- le rendu PyGame.

Une même fonction d'évaluation doit mesurer les deux agents avec les mêmes règles.

## Stratégie de `snake-algo.py`

Implémenter un planificateur `SafePath`, meilleur compromis score/temps qu'un Dijkstra ou un GBFS naïf :

1. Chercher un plus court chemin vers la pomme avec BFS ou Dijkstra. Sur une grille à coût uniforme, BFS donne le même plus court chemin avec moins de complexité de code ; conserver le nom et l'implémentation Dijkstra si sa présence est requise pour l'évaluation pédagogique.
2. Simuler le chemin candidat en tenant compte du déplacement de la queue et de la croissance après la pomme.
3. Accepter le chemin uniquement si, après la simulation, la tête peut encore rejoindre la queue ou si la zone libre accessible est assez grande pour le serpent.
4. Si le chemin est dangereux, suivre un chemin sûr vers la queue afin de gagner de l'espace.
5. Si aucun chemin complet n'existe, choisir le mouvement légal qui maximise la zone accessible par flood-fill, avec pénalité pour les culs-de-sac.

Ne jamais suivre aveuglément le chemin le plus court vers la pomme : il peut enfermer le serpent. Recalculer après chaque mouvement, car la queue et les obstacles changent.

Ordre des critères de départage : survie simulée, accès à la queue, espace libre, progression vers la pomme, coût de calcul.

## Stratégie de `snake-ia.py`

Construire un DQN PyTorch compact afin de satisfaire l'exigence Deep Learning sans créer une architecture trop longue à entraîner :

- état initial : danger devant/droite/gauche, direction courante en one-hot, position relative de la pomme ;
- réseau initial : MLP à deux couches cachées maximum ;
- sortie : trois actions relatives `tout droit`, `tourner à droite`, `tourner à gauche` ;
- adaptateur : convertir l'action relative en direction absolue comprise par le jeu ;
- apprentissage : replay buffer, mini-batch, politique epsilon-greedy, réseau cible et sauvegarde du meilleur checkpoint ;
- évaluation : epsilon nul et poids gelés.

Cette sortie relative évite de proposer directement un demi-tour illégal. Si l'enseignant exige quatre actions absolues, utiliser un masque d'actions et documenter ce changement.

Récompense de référence à reproduire d'abord : pomme `+10`, mort `-10`, fin du jeu `+100`, autres mouvements selon le support. Tester ensuite une faible pénalité par pas uniquement si la reward est modifiable, car `+0,1` par déplacement peut récompenser le fait de tourner longtemps au lieu de manger vite.

Ne pas présenter un modèle non entraîné comme un résultat. Charger explicitement un checkpoint pour toute démonstration.

## Protocole d'évaluation

Pour chaque agent, enregistrer au minimum :

- score final ;
- nombre de pommes ;
- nombre de pas ;
- durée de la partie ;
- temps moyen et maximum de décision ;
- cause de fin ;
- seed ;
- version/configuration de l'agent.

Comparer sur plusieurs seeds identiques. Publier moyenne, médiane, écart-type, meilleur score et taux de survie. Une seule meilleure partie ne suffit pas. Le critère principal recommandé est le nombre moyen de pommes avant collision ; le nombre de pas par pomme départage les agents aussi sûrs.

## Règles de travail pour l'IA de code

- Commencer par inspecter les interfaces et tests existants ; ne pas supposer les noms de classes ou fonctions.
- Faire des changements petits, testables et séparés.
- Ne jamais modifier silencieusement une contrainte du sujet.
- Ajouter un test pour chaque bug corrigé lorsque cela est raisonnable.
- Tester les cas : mur, corps, pomme adjacente, corridor, cul-de-sac, queue mobile, croissance et grille presque pleine.
- Fixer les seeds des bibliothèques utilisées pour les comparaisons.
- Garder les dépendances minimales : Python, PyGame existant et PyTorch pour le DQN.
- Ne pas ajouter de framework lourd sans bénéfice mesuré.
- Tenir à jour le changelog/timeline : date, auteur, changement, difficulté, résultat et preuve.
- Ne déclarer une amélioration qu'après comparaison sur le même protocole.

## Définition de terminé

Un livrable est terminé lorsqu'il :

- s'exécute depuis le dossier du groupe avec la commande documentée ;
- respecte les règles du jeu inchangées ;
- joue sans intervention humaine ;
- produit les métriques attendues ;
- passe les tests déterministes ;
- contient les auteurs et la timeline ;
- permet de reproduire au moins une évaluation ;
- pour le DQN, charge un modèle entraîné et démontre l'usage effectif du réseau.

## Questions ouvertes bloquant l'implémentation finale

- Quel est le dossier du groupe (`abricot` à `tomate`) et quels sont les auteurs/logins ?
- Quelles sont les échéances exactes ? Le README indique RL le 16/09 et algorithme le 22/09.
- Le reward d'entraînement peut-il différer des valeurs proposées dans le support ?
- L'entraînement sans rendu, plus rapide, est-il autorisé si l'évaluation conserve la clock ?
- Comment le professeur calcule-t-il officiellement le ratio score/temps ?
- Les actions relatives avec conversion vers les quatre directions sont-elles acceptées ?
- Quel matériel est disponible pour l'entraînement : CPU seulement ou GPU ?

