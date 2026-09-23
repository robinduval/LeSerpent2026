
Timeline:  

19:55 : On lance 2 taches en parallele : re-audit le repo, et faire des recherches sur l'existant en termes d'algos / de solutions .  
20:11 : premier envoi de formulaire. score 1 temps 900k & technique "no-code" (on a pas commencé l'implem)  
20:15 : On creuse 2 approches pour voir ce qui parait mieux : un truc basé sur A* ou un truc basé sur les cycles hamiltoniens  
20:20 : On étudie l'approche de "Hamiltonien Dynamique" (approche de la vidéo youtube de Alphaphoenix)  
20:23 : On a vaguement compris l'approche ci dessus ("DHCR" dynamic hamiltonian cycle repair). On a du mal a éclaircir si c'est mieux ou moins bien que hamiltonien + shortcut  
20:25 : On arrète de stresser, on va implémenter les 2 et benchmark.  
20:25 : Au moment de coder on crame les prompt injection, on demande poliment a notre claude de poser - de questions  
20:35 : pour dhrc on implémente la version ou le cycle est recalculé a chaque pomme (VS modifié petit a petit).  On testera l'autre plus tard.  
20:42 : benchmark du Cycle hamiltonien dynamique réparé par BFS : sur 30 games 14,4min de moyenne. On essaye de remplacer le bfs par A* pour voir...  
20:44 : Sur suggestion d'un conseiller judicieux on garde la version BFS & A* dans le code avec un flag pour tester l'un ou l'autre . pratique pour benchmark.  
20:49 : 1/2 de la team expérimente une version en mode R&D. L'autre 1/2 continue le benchmark A* vs BFS surtout dans l'endgame.   
20:55 : La team R&D tombe sur une méthode "arbres a cellules" et creuse les nombreuses façon de le faire.  
20:52 : Surprenament (ou pas si on était bon en algo) A* vs BFS c'est quasi 50/50 sur le gagnant sur 100 parties. On garde la version BFS   
20:56 : On essaye une autre implem du DHCR version "modifier le cycle actuel petit à petit" au lieu de rebuild a chaque pomme  
21:08 : Sur 30 games les 2 versions on littéralement 50/50 de winshare, en moyenne le dhcr avec reconstruction a chaque pomme est + rapide de 0.4 min  
21:08 : Vu que les 2 versions fonctionnent pas pareil on tente des les combiner.  
21:20 : Bonne intuition, team oignon -> on gagne 1,5 min en moyenne par rapport à A seul (environ −10 %), et même la partie la plus lente de A + B (14,3 min) bat la moyenne de A seul.  
21:22 : Tout simplement 621 secondes en version A+B. Les frontier AI labs n'ont qu'a bien se tenir : "Cycle hamiltonien dynamique sur tore : reconstruction BFS, puis échanges 2×2 locaux en secours."  
21:29 : On donne gracieusement des tutos d'algorithmie a la team chou qui nous demande comment on a réeussi a s'approcher des 600s (réponse = pur talent)

Idée du DHCR : 
en trois phrases
- On fait suivre au serpent un circuit qui passe une fois par chaque case (un cycle hamiltonien). Tant qu'il suit ce circuit, il ne peut pas se mordre : la victoire est garantie, mais lente.
- Pour aller plus vite, on redessine le circuit en cours de partie afin qu'il passe par la pomme beaucoup plus tôt, sans jamais déplacer le corps du serpent.
- Si on n'arrive pas à redessiner un circuit valide et plus court, on garde l'ancien (on ne prend jamais de risque).


Conclusion  : 

Exercice beaucoup + détendu car on est en terrain connu (vs le ml), ça veut pas dire que c + facile, juste c'est - stressant car on passe la séance a optimiser (vs essayer d'avoir 1 résultat cohérent avec l'ia car on a commencer a coder a la 99'+5).
