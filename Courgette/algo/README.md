
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

Cette méthode est rapide et fonctionne bien, mais elle ne garantit pas la
victoire. En fin de partie, le serpent peut créer une cavité dont il ne peut
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

Le test sur le parcours complet est encore en cours. Après environ 6 800 coups,
la pomme située dans la bande de la ligne 14 impose encore un détour de 15
coups.



## Autres solutions de fin de cours

On a demandé à l'IA de trouvé un solution adapté par la suite en echangeant avec elle 
pour le moment on trouve une option 14% plus rapide.

la nouvelle idée consiste en meatn en place le principe d'arbre : TODO 