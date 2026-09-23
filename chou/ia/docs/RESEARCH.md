# Recherche ciblée — Snake RL

Consultation effectuée le 16 septembre 2026. Ce document explique les décisions d'architecture et les pistes expérimentales. Les performances effectivement obtenues dans ce dépôt figurent dans `RESULTS.md`; les articles ne constituent pas des résultats du projet.

**Clarification utilisateur en cours de mission:** entraînement accéléré et modification des récompenses d'entraînement autorisés; évaluation du jeu conservée à 5 Hz. La carte blanche sur l'entraînement permet aussi d'explorer les démonstrations, en les déclarant. Cette autorisation explicite remplace l'interprétation prudente initiale tirée du silence du support. Elle n'autorise pas à changer le score officiel, la physique du jeu, les pommes ou la cadence d'évaluation. Les extensions de sécurité à l'exécution restent distinctes du réseau RL seul.

## Sources consultées et décisions

### 1. Double DQN — base retenue

H. van Hasselt, A. Guez, D. Silver, *Deep Reinforcement Learning with Double Q-learning* (2015/2016). [Notice](https://arxiv.org/abs/1509.06461), [texte, notamment section Double DQN](https://arxiv.org/html/1509.06461v3). Consultation de la notice et des sections de méthode; pas une reproduction de leurs expériences Atari.

Le réseau en ligne sélectionne l'action suivante; un réseau cible distinct évalue cette action. Cette séparation réduit une source de surestimation des valeurs Q. Application prévue: cible `r + gamma * Q_target(s_next, argmax Q_online(s_next))`, calculée sans gradients; aucun bootstrap après une vraie fin du jeu. Une interruption de budget n'est pas automatiquement une défaite. Réseau cible synchronisé explicitement, replay contenant des copies, exploration fondée sur le nombre de transitions et désactivée en évaluation.

Compatibilité: change l'apprentissage, pas les règles. Choix adapté à une première implémentation vérifiable et à une inférence CPU compacte. Le papier ne garantit ni la complétion de ce Snake ni l'optimalité selon son classement.

### 2. Prioritized Experience Replay — extension à isoler

T. Schaul et al., *Prioritized Experience Replay* (2015/2016). [Notice](https://arxiv.org/abs/1511.05952), [texte, sections 3.2 à 3.4 et algorithme 1](https://arxiv.org/html/1511.05952v4). Sections de méthode consultées.

PER réutilise plus souvent les transitions ayant une grande erreur TD. Application possible: priorités `abs(td_error) + epsilon`, probabilité proportionnelle à une puissance des priorités, poids d'importance normalisés et actualisation des priorités après apprentissage. Le correctif d'importance est nécessaire pour traiter le biais d'échantillonnage; il ne rend pas de nouvelles interactions disponibles.

Compatibilité: le même jeu et les mêmes observations restent employés. Intérêt particulier lorsque les interactions sont limitées par 5 Hz. Conserver un replay uniforme comme référence et comparer à budget d'interactions égal. Une meilleure loss seule ne suffit pas à retenir PER; trop réutiliser un petit buffer peut dégrader la généralisation. Aucun gain local n'est supposé avant mesure.

### 3. Rainbow — ablations et n-step, pas la pile entière

M. Hessel et al., *Rainbow: Combining Improvements in Deep Reinforcement Learning* (2017/2018). [Notice](https://arxiv.org/abs/1710.02298), [texte, multi-step et ablations](https://arxiv.org/html/1710.02298v1). Sections de méthode et d'analyse consultées.

L'article combine six améliorations et examine leur contribution séparément. Les retours multi-step propagent une récompense sur plusieurs transitions; ils modifient le compromis biais/variance. Application possible: tester `n=3` après `n=1`, avec somme actualisée correcte, facteur de bootstrap `gamma**k`, vidage des dernières transitions et séparation stricte des épisodes.

Décision: baseline, PER seul, n-step seul, puis combinaison si le budget le permet. Distributional RL, dueling et Noisy Nets sont différés: davantage de paramètres et de vérifications sans bénéfice établi ici. Les réglages Atari ne sont pas importés tels quels. La méthodologie d'ablation est retenue; aucune amélioration n'est annoncée à partir du seul nom Rainbow.

### 4. DQfD — extension d'entraînement possible, à déclarer

T. Hester et al., *Deep Q-learning from Demonstrations* (2017/2018). [Notice et abstract consultés](https://arxiv.org/abs/1704.03732). Le texte intégral n'a pas été analysé pour cette livraison.

L'abstract décrit l'association d'apprentissage TD, de classification supervisée des actions démontrées et de replay priorisé. Cela motive un préapprentissage suivi d'amélioration par interaction lorsque des démonstrations sont disponibles.

La clarification utilisateur autorise une large latitude d'entraînement. Une imitation seule ne prouverait pas l'apprentissage par renforcement demandé. Si cette extension est exploitée, mesurer séparément le coût de collecte, le préapprentissage et le RL; évaluer le réseau sans démonstrateur. Préférer des trajectoires de score élevé et de serpents longs plutôt qu'un expert simplement rapide. Il s'agit d'une piste autorisée, pas d'une composante validée par le présent document.

### 5. Shielding — distinction retenue, filtre différé

M. Alshiekh et al., *Safe Reinforcement Learning via Shielding* (2017/2018). [Notice et abstract consultés](https://arxiv.org/abs/1708.08611). Pas d'analyse intégrale des preuves formelles.

Le travail synthétise un mécanisme réactif à partir d'une spécification de sécurité et distingue filtrage avant choix et correction après choix. Il motive une séparation nette entre politique apprise et mécanisme de sécurité.

Décision: aucun filtre heuristique intégré par défaut tant que sa recevabilité n'est pas établie. Un détecteur de collision immédiate, un flood fill ou un chemin vers la queue ne donne pas les garanties formelles de cet article et ne prouve pas la possibilité de finir. Si un filtre est ajouté: journaliser action proposée et action exécutée, apprendre la transition exécutée, comparer réseau seul et système complet. Restreindre une interface aux trois actions relatives définies ne doit pas être confondu avec supprimer des actions définies mais dangereuses.

### 6. AlphaSnake — objectif et anticipation, infrastructure différée

K. Du et al., *AlphaSnake: Policy Iteration on a Nondeterministic NP-hard Markov Decision Process* (2022). [Notice](https://arxiv.org/abs/2211.09622), [texte, environnement, expériences et architecture](https://arxiv.org/html/2211.09622v1). Sections citées consultées.

L'article donne priorité à la probabilité de victoire, puis au nombre de pas pour gagner, et combine réseau appris et MCTS en traitant le placement des pommes comme un événement aléatoire. Ses plans d'entrée distinguent les directions des segments, la tête, la queue et la pomme: le corps ne se réduit pas à l'occupation des cases.

Retenu: anticiper la libération du corps et séparer réussite et vitesse. Différé: apprentissage avec MCTS, trop coûteux et non justifié avant un baseline mesuré. Les expériences utilisent une grille 10×10, une longueur initiale de 2 et une limite de 1 200 pas; aucune comparaison directe de score avec le moteur local. Une stratégie hamiltonienne exige une vérification sur la topologie réelle et resterait un algorithme programmé, pas à elle seule un agent RL.

### 7. Trust-Region Twisted Policy Improvement — piste de planification

J. A. de Vries et al., *Trust-Region Twisted Policy Improvement* (2025), version 4. [Notice](https://arxiv.org/abs/2504.06048), [texte, abstract et annexe expérimentale B](https://arxiv.org/html/2504.06048v4). Consultation ciblée de ces sections.

La méthode adapte la planification SMC à l'amélioration d'une politique, notamment par propositions d'actions contraintes, traitement des terminaisons et estimation des cibles. Elle est intéressante pour étudier le rapport qualité de décision/coût de recherche.

Différée: infrastructure de planification et dépendances non nécessaires au baseline. L'annexe utilise Jumanji Snake 12×12, récompense +1 par pomme, observations en cinq canaux, limite de 4 000 pas et collecte parallèle. Ce protocole ne remplace pas les règles locales. Toute future recherche locale devra avoir son propre aléatoire et mesurer sa latence à 5 Hz à l'évaluation. Aucun gain de temps officiel n'est déduit de leurs temps de calcul.

### 8. Twice Sequential Monte Carlo for Tree Search — piste avancée

Y. Oren et al., *Twice Sequential Monte Carlo for Tree Search* (2025, révision 3 du 21 mai 2026). [Notice](https://arxiv.org/abs/2511.14220), [texte, abstract et annexe D](https://arxiv.org/html/2511.14220v3). Consultation ciblée de ces sections.

TSMCTS cherche à réduire variance et dégénérescence des chemins de SMC quand la profondeur augmente, tout en conservant des propriétés favorables au calcul parallèle. L'étude utilise notamment Jumanji Snake et une infrastructure GPU avec environnements parallèles.

Différée: l'échelle des expériences et la complexité du planificateur ne correspondent pas au premier budget local. Si cette recherche est expérimentée, comparer plusieurs budgets sur les mêmes épisodes et séparer coût d'apprentissage, simulations et temps officiel. L'article fournit une piste pour mieux utiliser un budget de recherche, pas une preuve de complétion de ce moteur.

### 9. Dépôt pédagogique Patrick Loeber — origine, pas moteur de référence

[Dépôt et README](https://github.com/patrickloeber/snake-ai-pytorch), [agent.py](https://raw.githubusercontent.com/patrickloeber/snake-ai-pytorch/main/agent.py), [model.py](https://raw.githubusercontent.com/patrickloeber/snake-ai-pytorch/main/model.py), [game.py](https://raw.githubusercontent.com/patrickloeber/snake-ai-pytorch/main/game.py), consultés directement.

L'agent utilise onze caractéristiques, trois actions relatives et un réseau 11→256→3. Son exploration dépend du nombre de parties et il sauvegarde un record individuel; ces deux choix sont remplacés respectivement par un calendrier de transitions et une validation de plusieurs épisodes. [Code agent](https://raw.githubusercontent.com/patrickloeber/snake-ai-pytorch/main/agent.py).

Le trainer pédagogique utilise le même réseau pour prédiction et bootstrap, sans réseau cible distinct ni détachement explicite de la cible. Il ne constitue donc pas une implémentation Double DQN à reprendre telle quelle. [Code modèle](https://raw.githubusercontent.com/patrickloeber/snake-ai-pytorch/main/model.py).

Le moteur en ligne a des murs mortels, 40 Hz, une croissance immédiate et une limite `100*len(snake)`; ces choix ne sont pas ceux du moteur local. Ils ne sont pas importés. Le dépôt affiche une licence MIT; toute réutilisation de code devrait conserver l'attribution et la licence. [Code jeu](https://raw.githubusercontent.com/patrickloeber/snake-ai-pytorch/main/game.py), [dépôt](https://github.com/patrickloeber/snake-ai-pytorch).

## Application au moteur local et limites du raisonnement

Les observations suivantes sont des déductions de l'inspection locale, distinctes des résultats des articles. Le plateau est un tore 15×15, sans obstacles, et la croissance intervient au mouvement suivant. Les dangers doivent donc être calculés après prise en compte du passage aux bords et du retrait éventuel de la queue. L'indicateur de croissance en attente affecte réellement la prochaine transition.

Les onze informations pédagogiques sont une référence compacte, mais ne décrivent pas tout le corps: plusieurs configurations ayant la même observation peuvent nécessiter des décisions différentes. Une représentation enrichie devrait distinguer tête, pomme, direction, corps ordonné et croissance en attente. Le rang depuis la queue donne un ordre de libération, pas une échéance fixe lorsque le serpent grandit. La latitude d'entraînement clarifiée permet d'explorer cet enrichissement en conservant les onze informations comme comparaison. Ni prochains tirages de pommes ni état caché du générateur aléatoire ne font partie d'une observation légitime.

Le classement est lexicographique sur le **score officiel**, puis le **temps officiel à score identique**. Un retour RL actualisé n'est mathématiquement pas ce classement. Les récompenses d'entraînement peuvent maintenant être adaptées, mais chaque formule doit être déclarée et le score officiel rester inchangé. Sélectionner sur le score mesuré évite de rebaptiser la loss ou la durée de survie « performance officielle », sans prouver l'alignement parfait de l'objectif d'apprentissage. Une récompense positive de déplacement peut favoriser des boucles: les résultats doivent rendre cette stagnation visible. Toute pénalité temporelle d'entraînement reste un auxiliaire expérimental et ne devient jamais un barème de classement `score - coefficient * temps`.

À 5 transitions par seconde, 1 000 transitions demanderaient environ 200 secondes au minimum et 10 000 environ 33 minutes 20 secondes, hors pauses et surcoûts. L'autorisation explicite d'accélérer l'entraînement lève ce goulot d'étranglement pour la collecte; la dynamique, la grille et la distribution des pommes demeurent identiques. La validation officielle reste cadencée à 5 Hz. Les diagnostics accélérés sont identifiés comme tels et ne produisent pas des « temps officiels ». Réutiliser les transitions via replay augmente les mises à jour sans créer d'expérience supplémentaire; le ratio mises à jour/interactions et le temps d'entraînement doivent être enregistrés.

## Hypothèses à vérifier expérimentalement

1. Le petit DDQN apprend mieux qu'une politique non entraînée sur les mêmes observations, avec plusieurs graines d'évaluation et exploration coupée.
2. PER apporte un gain de score à nombre d'interactions égal, après vérification des poids d'importance et de la stabilité des valeurs Q.
3. Les retours n-step améliorent la propagation des pommes sans dégrader les transitions terminales; tester séparément n-step et PER.
4. L'information d'ordre du corps améliore les scores lorsque le serpent s'allonge; le prouver en comparaison contrôlée, pas par intuition seule.
5. Les écarts de score se reproduisent sur validation puis test final réservé. Documenter maximum, moyenne, médiane, dispersion, fréquence d'échec et atteinte d'une cible; aucune sélection sur une seule partie record.
6. Un gain de latence d'inférence améliore éventuellement le temps réel seulement s'il réduisait auparavant un dépassement de tick. Comparer les temps à score égal; ne jamais utiliser `score/temps`.

Ce sont des hypothèses et un ordre de priorités, pas une liste d'expériences prétendument terminées. Le statut réel de chaque variante, les budgets consommés et les limites statistiques doivent être lus dans `RESULTS.md`.

## Complément du sprint : cycles hamiltoniens

[John Tapsell, Nokia 6110 Part 3 – Algorithms](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/), consulté le 16 septembre 2026 : ordre du corps sur un cycle, raccourcis et réserve nécessaire à la croissance. L'adaptation locale conserve un corps contigu sur un cycle réorganisable ; elle est justifiée dans `HAMILTONIAN_SAFETY.md`. La sécurité est algorithmique et les décisions restantes sont apprises. Les performances de l'auteur ne sont pas transférées à notre jeu. La planification trop dominante a été écartée comme solution RL principale après mesure de l'influence réelle des poids.
