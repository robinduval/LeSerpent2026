19:00 : Arrivée du prof 
19:10 : Avec Bob, on se demande si l'usage de machin truc permettra de blabla alors nous allons essayer trucmuch...
19:20 : Finalement, ça marche pas, alors on va essayer de bidouiller le petit zinzin
19h50 : On se dit que un dijkstra, pour une solution optimal est insufisant car cela ne gere pas: le corp du serpent ni le tore plat
20h00: Creer un prompt pour savoir si l'optimal est raisonable / possible ou si on doit faire des heuristique pour avoir un resutlat satisfaisant dans un temps ou l'homme entre le chaise et la clavier reste en vie dans un temps raisonnable
20h10 : Definir ce qu'est l'optimal pour le jeu du snake c'est differents points: - Score maximal - Minimum de déplacements pour une séquence de pommes connue à l’avance - Minimum espéré pour des pommes uniformément aléatoires : politique optimale d’un processus de décision markovien.
Probabilité maximale de victoire : inutile ici si une stratégie déterministe garantit déjà 100 %.
20h30 : BFS torique temporisé : chaque case du corps porte sa date de libération, donc on traverse une case qu'on aura quittée. Pas de hamiltonien.
20h40 : Piège grow_pending : grow() est appelé après le move(). La case de la queue est libre, sauf si une croissance est en attente. 10 tests ciblés, 10/10.
20h50 : Accélération x40 avec chrono simulé (coups / 5) : le temps affiché ne dépend pas de la vitesse réelle.
21h00 : Mesure : en survie, se dérouler bat viser la pomme. 220 contre 185 à budget égal, et moins de coups à chaque jalon.
21h10 : Mesure : les contrôles d'aire font errer le serpent. aire050 finit 0/12 parties au-dessus de 220.
21h14 : On a bien spécifier nos specs on peut donner des premiers résultats
21h15 : Mesure : budget 15 / 60 / 150 ms donnent des résultats identiques. Le calcul n'est jamais le facteur limitant.
21h20 : Métrique retenue : score / temps en secondes, valide seulement si la partie finit au-dessus de 220.
21h30 : Solveur exact de fin de partie écrit puis retiré : aucun changement sur 12 parties à 3 seuils, et 2x plus lent.
21h35 : Bug des cases vides restantes : cycle de survie infini, 60 000 coups. Corrigé par mémoire des états visités depuis la dernière pomme.
21h40 : Résultat : victoire complète seed 6, score 223 en 1 208 s, ratio 0.185. Sur 12 parties : moyenne 0.165, meilleure 0.215, médiane de score 219.
