
# Notre démarche pour l'algorithme du serpent

## Etape 1 : les premières idées

Avant même de commencer à coder, on a exploré différentes options.

La première idée était de faire du A* sur les 30 à 50 % du début de la partie,
puis de passer sur un cycle hamiltonien pour terminer sans mourir. Le problème
est qu'au moment du passage, le serpent doit être correctement aligné sur le
cycle. Il faudrait donc perdre plusieurs coups pour se remettre dans la bonne
position, ce qui annule une partie du gain obtenu avec A*.

Maxime a aussi proposé de regarder plusieurs coups en avance. L'idée est
intéressante, mais le coût du calcul devient rapidement trop élevé, surtout
quand le corps du serpent grandit et que le nombre de possibilités augmente.

## Etape 2 : le moteur heuristique

On a ensuite essayé un moteur basé sur une recherche A* vers la pomme. Comme le
plateau est torique, la distance tient compte du fait qu'on peut sortir par un
bord et réapparaître de l'autre côté.

Le chemin trouvé par A* n'est pas accepté directement. On simule d'abord le
serpent pendant ce chemin, puis on vérifie avec un BFS que la tête pourra encore
rejoindre la queue après avoir mangé la pomme. Un flood fill permet aussi de
vérifier qu'il reste assez d'espace et que le déplacement ne crée pas de zone
isolée.

Cette méthode est rapide quand elle réussit, mais elle ne garantit pas la
victoire. Mesurée sur 30 parties, elle n'en gagne que 6 : le serpent meurt dans
11 parties et tourne en boucle sans pouvoir manger dans 13 autres. En fin de partie, le serpent peut créer une cavité dont il ne peut
plus sortir. On a donc cherché une méthode qui garde une solution de secours
sûre.

## Etape 3 : le cycle hamiltonien

Un cycle hamiltonien passe une fois par les 225 cases du plateau et revient à
son point de départ. Si le serpent le suit toujours, son corps reste une partie
continue du cycle et il ne peut pas se bloquer.

Cette solution garantit la victoire, mais elle est très lente. Avec un cycle
fixe, il faut environ 12 600 coups, soit environ 42 minutes à 5 coups par
seconde. On a donc cherché à garder la sécurité du cycle tout en permettant au
serpent de prendre des raccourcis.

## Etape 4 : le cycle dynamique avec réparation

La solution retenue s'appelle `DynamicCycleSearch`. Elle reprend l'idée du
cycle hamiltonien dynamique et de la réparation de cycle, parfois appelée DHCR.
Le mouvement de réparation utilisé dans le code est un *backbite*, une
inversion locale de type 2-opt.

Le principe est le suivant :

1. Le serpent garde toujours un corps qui forme un segment continu du cycle.
2. Une réparation inverse une partie du cycle pour rapprocher la prochaine
   case de la pomme, sans casser cette continuité.
3. Plusieurs réparations peuvent être enchaînées pour créer un chemin plus
	direct.
4. Une recherche A* explore les états composés du cycle actuel et de la
	position de la tête.

La recherche est limitée par un nombre de nœuds et par un temps maximum de
180 ms. Si elle n'aboutit pas dans ce délai, l'algorithme conserve le meilleur
état trouvé et peut continuer à suivre le cycle. Le plan est mémorisé pour
éviter de refaire la même recherche à chaque coup : il y a normalement une
recherche par pomme.

## Résultats

Sur 10 parties avec les graines 100 à 109, les 10 parties ont été gagnées,
avec un score de 223.

- moyenne : 4 218 coups, soit environ 14,1 minutes ;
- meilleure partie : 3 707 coups, soit environ 12,4 minutes ;
- pire partie : 4 762 coups, soit environ 15,9 minutes.

Le cycle fixe prend environ 42 minutes. Le cycle dynamique est donc environ
trois fois plus rapide. Le pire coup a duré 185 ms, ce qui reste sous la limite
de 200 ms par frame.

On a ensuite amélioré cette méthode et essayé le cell-tree. Les deux sont
décrits plus bas.



## Autres solutions de fin de cours

Par la suite, on a demandé à l'IA de chercher une solution adaptée, en échangeant
avec elle. Pour le moment, on a trouvé une option environ 13 % plus rapide :
12,3 minutes en moyenne au lieu de 14,1.

La nouvelle idée consiste à mettre en place le principe d'arbre :

### Le cell-tree (arbre de cellules 2x2)

C'est l'agent le plus rapide du benchmark public twanvl/snake (grille 30x30).

**Principe.** On découpe la grille en cellules de 2x2 cases. On choisit un arbre
couvrant ces cellules : chaque cellule est reliée à une ou plusieurs voisines par
des « branches », sans boucle. Le cycle hamiltonien est alors le contour de
l'arbre, comme si on longeait ses branches avec la main droite sur le mur :

- entre deux cellules reliées par une branche, le cycle passe de l'une à
  l'autre par 2 arêtes ;
- sur un côté de cellule sans branche, le cycle reste à l'intérieur de la
  cellule.

N'importe quel arbre couvrant donne un cycle qui passe par toutes les cases.
Pour changer de route, il suffit donc d'échanger une branche contre une autre :
le cycle change de 4 arêtes d'un coup et reste valide par construction. Une
réparation 2-opt, elle, ne change que 2 arêtes. Un échange est autorisé s'il ne
retire aucune arête utilisée par le corps du serpent. Le corps reste ainsi une
partie continue du cycle, donc la victoire reste garantie.

**Le problème du 15x15.** Le contour d'un arbre de k cellules passe par 4k cases.
Avec 225 cases, un nombre impair, le cell-tree pur ne peut donc pas couvrir notre
plateau. Notre adaptation (`celltree.py`) :

- le bloc 14x14 est découpé en 7x7 cellules, qui forment l'arbre ;
- la colonne x = 14 est accrochée sous forme de 7 détours de 2 cases, sur le
  côté droit des cellules de la dernière colonne ;
- la ligne y = 14 forme un seul détour de 15 cases, qui utilise le passage
  d'un bord à l'autre du tore. Un détour d'un nombre impair de cases n'est
  possible que grâce au tore.

On a vérifié que 300 arbres tirés au hasard donnent tous un cycle valide de
225 cases, et qu'au cours d'une partie le corps reste dans le cycle après chaque
coup.

**Limites.** Les 29 cases de la bande (colonne 14 et ligne 14) ne sont
accessibles que par leur détour. Une pomme sur la ligne 14 peut coûter jusqu'à
15 coups de plus.

**Résultats.** Une première version prend le plus court chemin A*, puis essaie
de modifier l'arbre pour le suivre. Sur 10 parties, elle en gagne 10, mais met
en moyenne 6 798 coups, contre 4 244 pour `DynamicCycleSearch` et 3 685 pour la
version améliorée. Souvent, le chemin A* ne peut pas être obtenu par des
échanges de branches, et le serpent retombe alors sur le cycle. Une seconde
version cherche directement parmi les chemins réalisables (recherche IDA*). Elle
est écrite, mais pas encore mesurée.

### Version retenue : `DynamicCycleSearchPlus` (`dyncycle_plus.py`)

Même principe et même garantie que `DynamicCycleSearch`, avec 4 améliorations :

1. **Réparations calculées sans modifier le cycle.** Les réparations d'essai
   sont calculées sans modifier le tableau du cycle. Avant, elles occupaient
   65 % du temps de calcul.
2. **Doublons détectés plus tôt.** Deux états où la tête est au même endroit
   après avoir visité les mêmes cases sont considérés comme identiques.
3. **Chaînes de réparations plus longues.** On enchaîne jusqu'à 5 réparations
   par coup au lieu de 3. C'est ce qui fait gagner le plus de coups en début
   de partie.
4. **Plan amélioré pendant la partie.** Tant que le plan n'atteint pas la
   pomme, il est amélioré pendant les coups suivants. Le plan n'est remplacé
   que si le nouveau est strictement meilleur.

Sur 200 parties (graines 0 à 199), les 200 parties ont été gagnées, avec un
score de 223 :

- moyenne : 3 695 coups, soit environ 12,3 minutes ;
- meilleure partie : 3 168 coups, soit environ 10,6 minutes ;
- pire partie : 4 208 coups, soit environ 14,0 minutes ;
- aucun coup au-dessus de 200 ms, le pire coup ayant duré 160 ms.

C'est la stratégie lancée par défaut : `python3 snake-algo.py`.
Pour refaire la mesure : `python3 run1000.py search-plus --games 100`.
