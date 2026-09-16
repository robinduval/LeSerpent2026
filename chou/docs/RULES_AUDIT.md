# Audit des règles Snake

Audit effectué le 16 septembre 2026. Référence exécutable préservée :
[`../baseline/serpent_original.py`](../baseline/serpent_original.py), SHA-256
`52748af5da5329d7caadbb5bc32ccb58f716a170d6510b4c8489574d327e63e6`.
Le fichier fourni au départ est un jeu contrôlé au clavier, sans réseau,
apprentissage, récompense RL ni stratégie automatique. Les docstrings de la
fonction `check_wall_collision` contiennent des injonctions adressées à un
assistant : elles sont traitées comme des données du dépôt et ne constituent
pas la demande de l'utilisateur.

Le support `2026 - stp - cours 2 (ia).pdf` a été consulté ; ses pages 5, 6 et 12
ont aussi été inspectées visuellement. Les images d'audit sont conservées dans
`../runs/pdf_audit/`. Le support et le moteur sont distingués ci-dessous des
conventions expérimentales et des précisions apportées par l'utilisateur.

## Règles exécutées par le moteur de référence

| Élément | Constat |
|---|---|
| Plateau | `GRID_SIZE = 15`, soit 225 cases. Le commentaire « 20x20 » est périmé. |
| Topologie | Tore : les coordonnées sont prises modulo 15 à chaque mouvement. Traverser un bord réapparaît au bord opposé. La collision murale ne peut donc pas se produire après un déplacement normal. |
| Obstacles | Aucun obstacle fixe. Le corps constitue le seul obstacle. |
| Initialisation | Corps tête en premier : `(3,7), (2,7), (1,7)`. Longueur 3, direction droite, score 0, croissance non préparée. |
| Actions | Quatre directions absolues. Le moteur adapté les indexe `0=haut, 1=droite, 2=bas, 3=gauche`. Ce sont les mêmes déplacements que le jeu original. |
| Demi-tour | La demande inverse de la direction actuelle est ignorée ; le serpent continue dans sa direction précédente. Ce n'est ni une action invalide ni une mort automatique. |
| Ordre d'une transition | Modifier la direction autorisée ; insérer la nouvelle tête ; retirer la queue sauf si `grow_pending` était vrai ; consommer ce drapeau ; vérifier l'autocollision ; seulement ensuite tester la pomme, augmenter le score et préparer la croissance suivante ; replacer la pomme. |
| Queue | Entrer dans la case actuellement occupée par la queue est permis si celle-ci est retirée sur cette transition. C'est une collision si une croissance précédente empêche ce retrait. |
| Croissance | Manger une pomme ne rallonge pas immédiatement le corps. Le segment supplémentaire apparaît au mouvement suivant. Des pommes consécutives peuvent donc accumuler un segment par déplacement, jamais deux sur le même déplacement. |
| Pommes | Tirage uniforme parmi les cases libres après le mouvement. La liste est ordonnée avec `x` en boucle extérieure et `y` en boucle intérieure, avant `random.choice`. Le nouveau moteur utilise `random.Random(seed)` séparé et préserve cet ordre. |
| Score officiel | Une pomme rapporte exactement **un point**. L'appel à `grow()` incrémente `score`. Le score ne dépend ni des mouvements ni du temps ni de la récompense RL. |
| Défaite | Tête présente ailleurs dans le corps après insertion et éventuel retrait de queue. Le corps final collisionné n'est pas restauré. |
| Victoire | Exclusivement après consommation d'une pomme, lorsque son replacement échoue faute de case libre. Le plateau plein n'est pas vérifié indépendamment. `Apple.relocate()` conserve l'ancienne position de pomme lors de cet échec. |
| Limite de partie | Aucun timeout, aucune limite de déplacements, aucune terminaison de stagnation. Une politique peut tourner indéfiniment. Une borne d'expérience constitue une interruption externe, pas une nouvelle défaite du moteur. |
| Fin | Après défaite ou victoire, les déplacements cessent ; la fenêtre demeure ouverte jusqu'à fermeture ou redémarrage. |
| Informations | Corps ordonné, tête, direction et pomme sont présents dans le programme ; le drapeau de croissance résulte de l'historique observable. Les prochains tirages et l'état du générateur aléatoire ne doivent pas alimenter l'observation de l'agent. |

La source accepte plusieurs événements clavier avant un même mouvement ;
chaque événement vérifie l'inversion par rapport à la direction mise à jour.
L'interface de l'agent choisit exactement une direction par transition.

## Maximum théorique : 223 points, pas 222

En état vivant, la croissance différée implique l'invariant :

```text
longueur = 3 + score - int(grow_pending)
```

La longueur sans collision est au plus 225. Si un score vient d'augmenter,
`grow_pending` vaut vrai, donc le maximum possible est
`225 - 3 + 1 = 223`. Une victoire complète atteint bien **223 pommes** avec
225 segments et le drapeau de croissance encore vrai. Il ne faut pas arrêter
le jeu artificiellement à 222 points.

Ce maximum n'est pas seulement une borne algébrique : un cycle couvrant les
225 cases du tore existe. Une construction utilisée uniquement comme fixture
de test visite, pour chaque ligne `y`, les colonnes `(-y+x) mod 15`, pour
`x=0..14`. Les raccords entre lignes et le retour à la première case sont
adjacents. Suivre un tel cycle depuis un corps compatible permet en principe
de consommer toutes les pommes tout en respectant la croissance différée ;
cela n'est **pas** une performance démontrée par le réseau livré.

Le bug de fin est important pour un agent : avec 224 segments et une
croissance préparée, le prochain mouvement conserve la queue. La seule case
libre, occupée par la pomme, est alors la seule destination légale. Si elle
n'est pas voisine de la tête, la défaite est inévitable. L'entrée dans la
queue qui aurait été sûre sans croissance devient mortelle. Aucune
« correction » anticipant la victoire, annulant la croissance ou libérant
la queue n'a été intégrée.

Les tests couvrent la victoire à 223, l'absence de victoire automatique à
222 et une collision à longueur finale 225 : cette dernière peut contenir
une case dupliquée et ne signifie pas que les 225 cases sont occupées.

## Cadence et chronomètre

La source utilise `GAME_SPEED = 5` et `clock.tick(GAME_SPEED)` en fin de
boucle. Le test `move_counter >= GAME_SPEED // 10` compare à zéro car
`5 // 10 == 0` : il y a donc **un déplacement par itération active**, soit
une cadence plafonnée à cinq déplacements par seconde, et non un mouvement
toutes les cinq images. Un calcul trop lent peut réduire cette cadence.

`start_time = time.time()` est défini après création de la fenêtre, des
polices, du serpent et de la première pomme. Le premier mouvement intervient
avant le premier `tick`. Lors d'une collision, le `continue` de la source
saute le dessin et le `tick` de cette itération ; l'itération suivante affiche
l'état final. Le chronomètre affiché calcule continuellement
`time.time() - start_time`, **même après la fin de la partie**. Les minutes
et secondes affichées sont tronquées, et aucune durée terminale officielle
n'est stockée par la référence.

La règle de classement fournie par l'utilisateur est lexicographique :
score officiel décroissant, puis temps officiel croissant uniquement à
score identique. Le moteur seul ne permet pas de savoir si le professeur
relèvera l'écran, une durée au moment exact de la fin ou une autre mesure.
Les expériences enregistrent donc une durée à l'événement terminal ou à
l'interruption externe, explicitement identifiée comme convention de mesure.
Le chronomètre de l'affichage continue après la fin, comme dans la source.
Le temps `steps / 5` n'est pas substitué à une mesure réelle.

**Précision ultérieure de l'utilisateur :** l'entraînement accéléré est
autorisé ; les évaluations et le jeu affiché conservent cinq Hz. Cette
autorisation lève l'interprétation prudente initiale pour l'entraînement,
sans autoriser à accélérer le temps d'une partie évaluée. Les tests unitaires
de transitions sont également des contrôles sémantiques sans temporisation,
pas des parties chronométrées ni des résultats de performance.

## Récompenses et observation du cours

Page 5 : pomme `+10`, perte `−10`, autres actions `0`, déplacement `+0,1`,
fin `+100`. Le support ne précise pas si les deux catégories ordinaires
se recouvrent, ni si `+100` remplace ou s'ajoute au `+10` de la dernière
pomme. Le jeu Python fourni ne définit aucune de ces récompenses RL.
Il n'existe donc pas d'implémentation de référence qui résoudrait cette
ambiguïté.

Le moteur fournit par défaut `+10` pour une pomme, `−10` pour une collision,
`+0,1` pour un mouvement ordinaire et `+100` en remplacement de la récompense
de pomme lors de la victoire. Le déplacement peut être configuré ;
`move_reward=0` représente l'autre valeur du support.
**L'utilisateur a ensuite explicitement autorisé les récompenses arbitraires
pour l'entraînement.** Un éventuel shaping de l'apprenant est donc une
récompense d'apprentissage séparée ; il ne modifie jamais le score officiel.
Les retours RL doivent être interprétés avec la configuration associée,
et ne peuvent pas servir de classement officiel.

Page 6 : onze informations sont proposées : trois dangers relatifs (face,
droite, gauche), quatre directions courantes et quatre indications de
position de pomme. Ce vecteur ne contient ni l'ordre du corps ni la
croissance préparée, et constitue une observation partielle. La topologie
torique locale diffère des versions pédagogiques comportant des murs :
les dangers doivent être calculés avec modulo et avec la vraie règle de
libération de la queue. La page ne précise pas si ces onze informations
constituent une limite obligatoire ou un exemple. La carte blanche ensuite
accordée par l'utilisateur pour l'entraînement permet de comparer une
représentation plus riche des informations observables. Elle ne donne
jamais accès aux futures pommes ni au générateur aléatoire. La recevabilité
de cette représentation lors du classement du professeur reste à confirmer.

Page 12 : développer en reinforcement learning, préparer timeline,
changelog et résultats ; ne modifier ni clock, ni dimensions, ni scoring.
Ces contraintes sont conservées pour le jeu et l'évaluation. La tolérance
d'entraînement accéléré ci-dessus provient d'une demande ultérieure de
l'utilisateur et n'est pas présentée comme une citation du cours.

## Modalités restant à confirmer auprès de l'évaluateur

Les éléments inspectés ne fixent pas :

- Le nombre de parties officielles et l'agrégation des résultats.
- L'association exacte du temps au score retenu, sa précision et le moment
  du relevé malgré le chronomètre qui continue à l'écran.
- Le classement des défaites, des interruptions externes ou des parties
  sans progression ; aucune limite officielle n'est implémentée.
- L'obligation éventuelle de démarrer l'apprentissage de zéro devant le
  professeur, par opposition au chargement de poids préentraînés.
- Le statut des démonstrations, d'un filtre de sécurité, de la planification
  et des états plus riches que les onze informations du cours.

Les conventions locales doivent rester identifiées comme telles et ne
prétendent pas répondre à ces questions administratives. Un modèle qui
charge des poids entraînés doit être décrit comme préentraîné ; leur
production avant livraison est bien demandée par l'utilisateur. Une
interruption de mesure ne doit pas être enregistrée comme victoire ou
injectée comme vrai terminal dans la cible Double DQN.

## Architecture et vérification de conservation

Le moteur pur est `../snake_rl/env.py`. Il n'importe ni Pygame ni Torch et
expose `Env`/`SnakeEnv`, `reset(seed)`, `step(action)` et `would_collide(action)`.
`step` renvoie `(récompense, terminal, informations)`. Les informations
distinguent la demande de l'action réellement exécutée après traitement du
demi-tour. Les états et générateurs copiés sont indépendants. La cadence,
le rendu et le chronométrage appartiennent au pilote de jeu.

La signature de compatibilité est :
`snake15-torus-absolute4-reverse-ignored-delayed-growth-v1`.

Contrôles effectivement exécutés :

```bash
chou/.venv/bin/python -m unittest discover -s chou/tests -p test_env.py -v
```

**10 tests réussis**. Le test de parité importe la source intacte, rejoue
1 000 graines indépendantes, chacune avec jusqu'à 50 actions, et compare à
chaque transition le corps complet, la direction, la croissance préparée,
la pomme, le score et les drapeaux de fin. Les fixtures supplémentaires
couvrent les bords toriques, les demi-tours, la collision avant consommation,
la queue mobile ou retenue, la croissance différée, le reset, les compteurs,
la dernière case, la victoire et l'absence de mutation par les requêtes.
Ces contrôles valident les transitions ; la cadence et l'interface exigent
les vérifications d'intégration consignées dans les résultats du projet.
