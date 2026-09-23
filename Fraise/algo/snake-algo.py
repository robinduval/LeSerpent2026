"""Snake piloté par des algorithmes de recherche de chemin (groupe Fraise).

Trois algorithmes, dans l'ordre du cours :
  - dijkstra : plus court chemin vers la pomme (coût uniforme)
  - gbfs     : Greedy Best-First Search (heuristique de Manhattan sur le tore)
  - hybride  : le nôtre (voir README.md)

Le jeu de base (serpent-algo.py) est importé tel quel : clock, grille et scoring d'origine.
"""

import argparse
import heapq
import random
import importlib.util
import os
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

# --- CHARGEMENT DU JEU DE BASE (SANS LE MODIFIER) ---
_chemin_jeu = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serpent-algo.py")
_spec = importlib.util.spec_from_file_location("serpent", _chemin_jeu)
serpent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serpent)

G = serpent.GRID_SIZE
DIRECTIONS = [serpent.RIGHT, serpent.DOWN, serpent.LEFT, serpent.UP]


# --- OUTILS DE GRILLE (les bords se traversent : c'est un tore) ---

def voisins(case):
    x, y = case
    return [((x + dx) % G, (y + dy) % G) for dx, dy in DIRECTIONS]


def pas_vers(depart, arrivee):
    """Direction à prendre pour aller d'une case à une case voisine (bords compris)."""
    dx = (arrivee[0] - depart[0]) % G
    dy = (arrivee[1] - depart[1]) % G
    if dx == 1:
        return serpent.RIGHT
    if dx == G - 1:
        return serpent.LEFT
    if dy == 1:
        return serpent.DOWN
    return serpent.UP


def distance_manhattan(a, b):
    """Distance sur le tore : passer par le bord peut être plus court."""
    dx = (a[0] - b[0]) % G
    dy = (a[1] - b[1]) % G
    return min(dx, G - dx) + min(dy, G - dy)


def libre_au_coup(corps, grow_pending):
    """Coup à partir duquel chaque case du corps est libre.
    Le segment n°i (tête = 0) quitte sa case après (longueur - i) coups,
    un coup de plus si le serpent vient de manger."""
    longueur = len(corps)
    retard = 1 if grow_pending else 0
    return {tuple(segment): longueur - i + retard for i, segment in enumerate(corps)}


# --- LE JEU, PILOTÉ PAR UN ALGORITHME ---

class SnakeGameAlgo:
    """Enveloppe le jeu de base pour qu'un algorithme puisse le piloter."""

    def __init__(self, affichage=True, facteur_vitesse=1):
        self.affichage = affichage
        # Mode rapide : la partie tourne facteur_vitesse fois plus vite,
        # et le temps compté est multiplié d'autant (x10 en vitesse = x10 sur le temps)
        self.facteur_vitesse = facteur_vitesse
        if self.affichage:
            pygame.init()
            self.screen = pygame.display.set_mode((serpent.SCREEN_WIDTH, serpent.SCREEN_HEIGHT))
            titre = "Snake Algo (Fraise)"
            if facteur_vitesse != 1:
                titre += f" - MODE RAPIDE x{facteur_vitesse} (temps affiché = temps réel x{facteur_vitesse})"
            pygame.display.set_caption(titre)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 40)
        self.reset()

    def reset(self):
        self.snake = serpent.Snake()
        self.apple = serpent.Apple(self.snake.body)
        self.victoire = False
        self.coups = 0
        self.coups_sans_pomme = 0
        self.start_time = time.time()

    def temps_ecoule(self):
        return (time.time() - self.start_time) * self.facteur_vitesse

    def tete(self):
        return tuple(self.snake.head_pos)

    def pomme(self):
        return tuple(self.apple.position)

    def play_step(self, direction):
        """Joue un coup dans une direction absolue. Retourne (game_over, score)."""
        if self.affichage:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        self.snake.set_direction(direction)
        self.snake.move()
        self.coups += 1
        self.coups_sans_pomme += 1

        # Mêmes vérifications que la boucle principale du jeu de base
        if self.snake.is_game_over():
            return True, self.snake.score

        if self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            self.coups_sans_pomme = 0
            if not self.apple.relocate(self.snake.body):
                self.victoire = True
                return True, self.snake.score

        if self.affichage:
            self._dessiner()
            if self.facteur_vitesse == 1:
                self.clock.tick(serpent.GAME_SPEED)  # clock d'origine, comme le jeu de base
            else:
                # même fréquence x facteur, mais attente précise : tick() perd ~3 ms par image,
                # ce qui serait multiplié par le facteur dans le temps compté
                self.clock.tick_busy_loop(serpent.GAME_SPEED * self.facteur_vitesse)

        return False, self.snake.score

    def _dessiner(self):
        self.screen.fill(serpent.GRIS_FOND)
        zone_jeu = pygame.Rect(0, serpent.SCORE_PANEL_HEIGHT, serpent.SCREEN_WIDTH, serpent.SCREEN_WIDTH)
        pygame.draw.rect(self.screen, serpent.NOIR, zone_jeu)
        serpent.draw_grid(self.screen)
        self.apple.draw(self.screen)
        self.snake.draw(self.screen)
        # display_info affiche (maintenant - début) : on lui passe un début qui donne le temps compté
        serpent.display_info(self.screen, self.font, self.snake, time.time() - self.temps_ecoule())
        pygame.display.flip()


# --- ALGORITHMES ---

class Algorithme:
    """Base commune : compte les cases explorées, pour comparer les algorithmes."""

    nom = "base"

    def __init__(self):
        self.cases_explorees = 0

    def choisir(self, game):
        raise NotImplementedError

    def _obstacles(self, game):
        """Cases interdites au prochain coup : le corps, sauf la queue qui avance."""
        occupation = libre_au_coup(game.snake.body, game.snake.grow_pending)
        return {case for case, coup in occupation.items() if coup > 1}

    def _premier_pas(self, chemin, game):
        """Direction du premier pas d'un chemin, ou None si pas de chemin."""
        if chemin:
            return pas_vers(game.tete(), chemin[0])
        return None

    def _survivre(self, game, obstacles):
        """Aucun chemin vers la pomme : on prend une case libre (sinon on meurt)."""
        for direction in DIRECTIONS:
            case = ((game.tete()[0] + direction[0]) % G, (game.tete()[1] + direction[1]) % G)
            if case not in obstacles and direction != (-game.snake.direction[0], -game.snake.direction[1]):
                return direction
        return game.snake.direction


def reconstruire(precedents, cible):
    chemin = [cible]
    while precedents[chemin[-1]] is not None:
        chemin.append(precedents[chemin[-1]])
    chemin.pop()  # on retire la case de départ (la tête)
    chemin.reverse()
    return chemin


class Dijkstra(Algorithme):
    """Plus court chemin vers la pomme : explore d'abord les cases les moins coûteuses.
    Chaque déplacement coûte 1, donc le chemin trouvé est optimal en nombre de cases."""

    nom = "dijkstra"

    def chemin(self, depart, cible, obstacles):
        file = [(0, depart)]
        couts = {depart: 0}
        precedents = {depart: None}
        explorees = 0
        while file:
            cout, case = heapq.heappop(file)
            if cout > couts.get(case, float("inf")):
                continue
            explorees += 1
            if case == cible:
                self.cases_explorees += explorees
                return reconstruire(precedents, cible)
            for voisin in voisins(case):
                if voisin in obstacles:
                    continue
                nouveau = cout + 1
                if nouveau < couts.get(voisin, float("inf")):
                    couts[voisin] = nouveau
                    precedents[voisin] = case
                    heapq.heappush(file, (nouveau, voisin))
        self.cases_explorees += explorees
        return None

    def choisir(self, game):
        obstacles = self._obstacles(game)
        chemin = self.chemin(game.tete(), game.pomme(), obstacles)
        return self._premier_pas(chemin, game) or self._survivre(game, obstacles)


class GBFS(Algorithme):
    """Greedy Best-First Search : va toujours vers la case qui semble la plus proche
    de la pomme (distance de Manhattan). Rapide, mais le chemin n'est pas optimal."""

    nom = "gbfs"

    def chemin(self, depart, cible, obstacles):
        file = [(distance_manhattan(depart, cible), depart)]
        vus = {depart}
        precedents = {depart: None}
        explorees = 0
        while file:
            _, case = heapq.heappop(file)
            explorees += 1
            if case == cible:
                self.cases_explorees += explorees
                return reconstruire(precedents, cible)
            for voisin in voisins(case):
                if voisin in obstacles or voisin in vus:
                    continue
                vus.add(voisin)
                precedents[voisin] = case
                heapq.heappush(file, (distance_manhattan(voisin, cible), voisin))
        self.cases_explorees += explorees
        return None

    def choisir(self, game):
        obstacles = self._obstacles(game)
        chemin = self.chemin(game.tete(), game.pomme(), obstacles)
        return self._premier_pas(chemin, game) or self._survivre(game, obstacles)


def accessible(depart, cible, corps, grow_pending, coup_depart=1):
    """Peut-on aller de depart à cible sans se cogner, sachant que la queue avance ?
    Propagation couche par couche : une case du corps devient franchissable
    quand son segment l'a quittée."""
    occupation = libre_au_coup(corps, grow_pending)
    # depart est la case où se trouve la tête : on y est déjà, elle ne bloque pas
    vus = {depart}
    frontiere = [depart]
    coup = coup_depart
    while frontiere:
        if cible in vus:
            return True
        coup += 1
        suivante = []
        for case in frontiere:
            for voisin in voisins(case):
                if voisin not in vus and occupation.get(voisin, 0) <= coup:
                    vus.add(voisin)
                    suivante.append(voisin)
        frontiere = suivante
    return cible in vus


class GloutonSur(Algorithme):
    """Plus court chemin vers la pomme (Dijkstra), mais on ne s'y engage que si,
    une fois la pomme mangée, la tête peut encore rejoindre sa queue.
    Sinon on suit sa queue en attendant que la place se libère."""

    nom = "sur"

    def __init__(self):
        super().__init__()
        self.dijkstra = Dijkstra()

    def _corps_apres(self, game, chemin):
        """Corps du serpent après avoir suivi le chemin jusqu'à la pomme (il grandit d'une case)."""
        corps = [tuple(segment) for segment in game.snake.body]
        for i, case in enumerate(chemin):
            corps.insert(0, case)
            if i < len(chemin) - 1:   # au dernier pas le serpent mange, donc il ne perd pas sa queue
                corps.pop()
        return corps

    def choisir(self, game):
        obstacles = self._obstacles(game)
        chemin = self.dijkstra.chemin(game.tete(), game.pomme(), obstacles)
        self.cases_explorees += self.dijkstra.cases_explorees
        self.dijkstra.cases_explorees = 0

        if chemin:
            corps_futur = self._corps_apres(game, chemin)
            if accessible(corps_futur[0], corps_futur[-1], corps_futur, True):
                return self._premier_pas(chemin, game)

        # Pas de chemin sûr vers la pomme : on suit sa queue
        queue = tuple(game.snake.body[-1])
        obstacles_queue = obstacles - {queue}
        chemin_queue = self.dijkstra.chemin(game.tete(), queue, obstacles_queue)
        self.cases_explorees += self.dijkstra.cases_explorees
        self.dijkstra.cases_explorees = 0
        if chemin_queue:
            return self._premier_pas(chemin_queue, game)
        return self._survivre(game, obstacles)


def construire_cycle():
    """Circuit hamiltonien sur le tore : chaque ligne est parcourue vers la droite en
    traversant le bord, puis on descend. Après 15 lignes on retombe sur la case de départ.
    Impossible sur une grille fermée 15x15 (225 cases, nombre impair), possible ici
    justement parce que les bords se rejoignent."""
    cycle, x, y = [], 0, 0
    for _ in range(G):
        for _ in range(G):
            cycle.append((x, y))
            x = (x + 1) % G
        x = (x - 1) % G  # on annule le dernier pas à droite
        y = (y + 1) % G  # et on descend
    return cycle


class Hybride(Algorithme):
    """Le nôtre : un circuit hamiltonien sert de filet de sécurité, et on le raccourcit
    dès que c'est sûr. Règle de sûreté : un raccourci ne doit jamais dépasser la queue
    dans l'ordre du circuit, donc la tête reste toujours devant le corps."""

    nom = "hybride"
    # Réglages retenus après mesures (voir README) : marge 1 a fini par mourir (60 parties),
    # marge 2 et 3 gagnent 60/60 ; seuil 0,45 est le plus rapide.
    MARGE = 3              # un raccourci s'arrête au moins 3 cases avant la queue
    SEUIL_PRUDENCE = 0.45  # sous ce taux de cases libres, on suit le circuit sans raccourci

    def __init__(self, marge=None, seuil=None, critere="cycle"):
        super().__init__()
        self.marge = self.MARGE if marge is None else marge
        self.seuil = self.SEUIL_PRUDENCE if seuil is None else seuil
        self.critere = critere  # "cycle" ou "manhattan" : comment départager deux raccourcis
        self.cycle = construire_cycle()
        self.index = {case: i for i, case in enumerate(self.cycle)}
        self.n = len(self.cycle)

    def _distance_cycle(self, depart, arrivee):
        return (self.index[arrivee] - self.index[depart]) % self.n

    def choisir(self, game):
        tete, queue, pomme = game.tete(), tuple(game.snake.body[-1]), game.pomme()
        obstacles = self._obstacles(game)
        vides = self.n - len(game.snake.body)

        saut_max = self._distance_cycle(tete, queue) - self.marge
        if vides < self.n * self.seuil:
            saut_max = 1  # fin de partie : plus de raccourci, on suit le circuit

        meilleur, meilleur_score = None, None
        for direction in DIRECTIONS:
            if direction == (-game.snake.direction[0], -game.snake.direction[1]):
                continue
            case = ((tete[0] + direction[0]) % G, (tete[1] + direction[1]) % G)
            if case in obstacles:
                continue
            self.cases_explorees += 1
            saut = self._distance_cycle(tete, case)
            if saut < 1 or saut > max(1, saut_max):
                continue
            # à saut autorisé, on préfère la case la plus proche de la pomme dans le circuit
            if self.critere == "manhattan":
                note = (distance_manhattan(case, pomme), -saut)
            elif self.critere == "saut":      # on avance le plus loin possible dans le circuit
                note = (-saut, self._distance_cycle(case, pomme))
            else:
                note = (self._distance_cycle(case, pomme), -saut)
            if meilleur_score is None or note < meilleur_score:
                meilleur, meilleur_score = direction, note

        if meilleur is not None:
            return meilleur
        suivante = self.cycle[(self.index[tete] + 1) % self.n]
        if suivante not in obstacles:
            return pas_vers(tete, suivante)
        return self._survivre(game, obstacles)


def zone_libre(depart, corps, grow_pending, optimiste=True):
    """Nombre de cases atteignables depuis depart.
    optimiste=True : on tient compte de la queue qui avance et libère des cases (vue large).
    optimiste=False : le corps est un mur (vue prudente) ; c'est ce qu'il faut pour détecter
    qu'on est en train de s'enfermer dans une poche."""
    if not optimiste:
        murs = {tuple(segment) for segment in corps[:-1]}
        vus = {depart}
        frontiere = [depart]
        while frontiere:
            suivante = []
            for case in frontiere:
                for voisin in voisins(case):
                    if voisin not in vus and voisin not in murs:
                        vus.add(voisin)
                        suivante.append(voisin)
            frontiere = suivante
        return len(vus)
    occupation = libre_au_coup(corps, grow_pending)
    vus = {depart}
    frontiere = [depart]
    coup = 1
    while frontiere:
        coup += 1
        suivante = []
        for case in frontiere:
            for voisin in voisins(case):
                if voisin not in vus and occupation.get(voisin, 0) <= coup:
                    vus.add(voisin)
                    suivante.append(voisin)
        frontiere = suivante
    return len(vus)


def corps_apres_pas(corps, case, mange):
    """Corps après un déplacement de la tête vers case (la queue reste si le serpent mange)."""
    nouveau = [case] + list(corps)
    if not mange:
        nouveau.pop()
    return nouveau


class Glouton(Algorithme):
    """Sans circuit : on fonce vers la pomme tant que c'est sûr, sinon on gagne du temps.

    Deux garde-fous :
      - on ne s'engage vers la pomme que si, après l'avoir mangée, la tête peut encore
        rejoindre sa queue (le corps est vu comme mouvant : la queue libère des cases) ;
      - passé SEUIL_LONGUEUR de la grille, le serpent est trop long pour prendre des risques :
        il ne vise plus la pomme et se contente de garder le plus d'espace possible.
    Plus rapide que le circuit hamiltonien, mais il ne gagne pas à tous les coups.
    """

    nom = "glouton"
    SEUIL_LONGUEUR = 0.6  # au-delà de cette part de la grille, on ne vise plus la pomme
    PATIENCE = 0          # 0 = on ne retente jamais la pomme passé le seuil

    def __init__(self, seuil_longueur=None, pomme_proche=0, survie="loin_queue", patience=None):
        super().__init__()
        self.seuil_longueur = self.SEUIL_LONGUEUR if seuil_longueur is None else seuil_longueur
        # même passé le seuil, on mange la pomme si elle est à moins de pomme_proche cases
        self.pomme_proche = pomme_proche
        # comment départager deux coups de survie à espace égal
        self.survie = survie  # "loin_queue", "pres_queue" ou "pres_pomme"
        # si aucune pomme depuis patience coups, on retente malgré le seuil (0 = jamais)
        self.patience = self.PATIENCE if patience is None else patience
        self.dijkstra = Dijkstra()

    def _chemin(self, depart, cible, obstacles):
        chemin = self.dijkstra.chemin(depart, cible, obstacles)
        self.cases_explorees += self.dijkstra.cases_explorees
        self.dijkstra.cases_explorees = 0
        return chemin

    def _sur_apres_chemin(self, game, chemin):
        """Après avoir suivi le chemin et mangé la pomme : peut-on encore rejoindre sa queue ?"""
        corps = [tuple(segment) for segment in game.snake.body]
        for i, case in enumerate(chemin):
            corps = corps_apres_pas(corps, case, mange=(i == len(chemin) - 1))
        return accessible(corps[0], corps[-1], corps, True)

    def choisir(self, game):
        obstacles = self._obstacles(game)
        corps = [tuple(segment) for segment in game.snake.body]

        prudent = len(corps) > self.seuil_longueur * G * G
        if self.patience and game.coups_sans_pomme > self.patience:
            prudent = False  # on tourne en rond depuis trop longtemps : on retente la pomme
        chemin = None if (prudent and not self.pomme_proche) else self._chemin(game.tete(), game.pomme(), obstacles)
        if chemin and (not prudent or len(chemin) <= self.pomme_proche) and self._sur_apres_chemin(game, chemin):
            return self._premier_pas(chemin, game)

        # Sinon : survivre en gardant le plus d'espace, et le plus loin possible de sa queue
        meilleur, meilleure_note = None, None
        for direction in DIRECTIONS:
            if direction == (-game.snake.direction[0], -game.snake.direction[1]):
                continue
            case = ((game.tete()[0] + direction[0]) % G, (game.tete()[1] + direction[1]) % G)
            if case in obstacles:
                continue
            futur = corps_apres_pas(corps, case, mange=(case == game.pomme()))
            espace = zone_libre(case, futur, case == game.pomme())
            self.cases_explorees += espace
            if self.survie == "pres_queue":
                note = (espace, -distance_manhattan(case, futur[-1]))
            elif self.survie == "pres_pomme":
                note = (espace, -distance_manhattan(case, game.pomme()))
            else:
                note = (espace, distance_manhattan(case, futur[-1]))
            if meilleure_note is None or note > meilleure_note:
                meilleur, meilleure_note = direction, note
        return meilleur or self._survivre(game, obstacles)


def poches_libres(corps):
    """Découpe les cases libres en poches séparées (le corps est vu comme un mur, queue exclue).
    Retourne la liste des poches, chacune étant un ensemble de cases."""
    murs = {tuple(segment) for segment in corps[:-1]}
    restantes = {(x, y) for x in range(G) for y in range(G)} - murs
    poches = []
    while restantes:
        depart = restantes.pop()
        poche = {depart}
        a_voir = [depart]
        while a_voir:
            case = a_voir.pop()
            for voisin in voisins(case):
                if voisin in restantes:
                    restantes.discard(voisin)
                    poche.add(voisin)
                    a_voir.append(voisin)
        poches.append(poche)
    return poches


class Audacieux(Algorithme):
    """Taillé pour la vitesse : on va toujours au plus court vers la pomme, avec un contrôle
    de sécurité plus ou moins léger. Il gagne rarement, mais quand il gagne, c'est vite.
    On compense en jouant des milliers de parties (voir chercher.py).

    controle :
      - "aucun"  : on fonce, on ne vérifie rien (le plus rapide, le plus mortel)
      - "espace" : on refuse un coup si l'espace atteignable tombe sous risque x longueur
      - "queue"  : on ne s'engage que si la queue reste joignable après avoir mangé
    """

    nom = "audacieux"

    def __init__(self, controle="espace", risque=0.5, rangement=False, seuil_rangement=0.55, survie="espace"):
        super().__init__()
        self.controle = controle
        self.risque = risque
        # rangement : passé seuil_rangement de la grille, on évite de laisser des poches
        # isolées derrière soi, pour que les pommes tombent devant le serpent et non dedans
        self.rangement = rangement
        self.seuil_rangement = seuil_rangement
        # survie : "espace" (le plus d'espace), "mur" (longer les murs pour ne pas couper
        # l'espace en deux), "loin_queue" (s'éloigner de sa queue)
        self.survie = survie
        self.dijkstra = Dijkstra()

    def _chemin(self, depart, cible, obstacles):
        chemin = self.dijkstra.chemin(depart, cible, obstacles)
        self.cases_explorees += self.dijkstra.cases_explorees
        self.dijkstra.cases_explorees = 0
        return chemin

    def _acceptable(self, game, chemin, corps, obstacles):
        if self.controle == "aucun":
            return True
        if self.controle == "queue":
            futur = corps
            for i, case in enumerate(chemin):
                futur = corps_apres_pas(futur, case, mange=(i == len(chemin) - 1))
            return accessible(futur[0], futur[-1], futur, True)
        # "espace" : contrôle léger, sur le premier pas seulement
        case = chemin[0]
        futur = corps_apres_pas(corps, case, mange=(case == game.pomme()))
        return zone_libre(case, futur, case == game.pomme()) >= self.risque * len(futur)

    def choisir(self, game):
        obstacles = self._obstacles(game)
        corps = [tuple(segment) for segment in game.snake.body]
        chemin = self._chemin(game.tete(), game.pomme(), obstacles)
        if chemin and self._acceptable(game, chemin, corps, obstacles):
            return self._premier_pas(chemin, game)

        # sinon : coup de survie
        ranger = self.rangement and len(corps) > self.seuil_rangement * G * G
        meilleur, meilleure_note = None, None
        for direction in DIRECTIONS:
            if direction == (-game.snake.direction[0], -game.snake.direction[1]):
                continue
            case = ((game.tete()[0] + direction[0]) % G, (game.tete()[1] + direction[1]) % G)
            if case in obstacles:
                continue
            futur = corps_apres_pas(corps, case, mange=(case == game.pomme()))
            espace = zone_libre(case, futur, case == game.pomme())
            self.cases_explorees += espace
            if ranger:
                # on veut le moins de poches séparées possible, et la plus grande sous la tête :
                # ainsi les prochaines pommes tombent devant le serpent et non dans un trou
                poches = poches_libres(futur)
                sous_la_tete = max((len(poche) for poche in poches if case in poche), default=0)
                # d'abord une grande zone sous la tête, ensuite le moins de poches isolées
                note = (sous_la_tete, -len(poches), espace)
            elif self.survie == "mur":
                # longer le corps ou le bord : moins on a de voisins libres, moins on coupe l'espace
                libres_autour = sum(1 for v in voisins(case) if v not in set(futur[:-1]))
                note = (espace, -libres_autour, distance_manhattan(case, futur[-1]))
            elif self.survie == "loin_queue":
                note = (espace, distance_manhattan(case, futur[-1]))
            else:
                note = (espace,)
            if meilleure_note is None or note > meilleure_note:
                meilleur, meilleure_note = direction, note
        return meilleur or self._survivre(game, obstacles)


ALGORITHMES = {classe.nom: classe for classe in (Dijkstra, GBFS, GloutonSur, Hybride, Glouton, Audacieux)}


# --- LANCEMENT DES PARTIES ---

# Au-delà de ce nombre de coups sans pomme, la partie tourne en boucle sans fin
LIMITE_BOUCLE = 3000


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def format_temps(secondes):
    return f"{int(secondes // 60):02d}:{secondes % 60:04.1f}"


def jouer_partie(game, algo, journal=None, numero=1, graine=None):
    """Joue une partie entière. Retourne (score, temps de jeu, fin, coups).
    graine : fixe le tirage des pommes, donc la partie est exactement reproductible."""
    if graine is not None:
        random.seed(graine)
    game.reset()
    algo.cases_explorees = 0
    score, fini = 0, False
    score_precedent = 0
    while not fini:
        direction = algo.choisir(game)
        fini, score = game.play_step(direction)
        if journal and score > score_precedent:
            journal(f"Partie {numero} | Score {score:3d} | Temps {format_temps(game.temps_ecoule())}")
            score_precedent = score
        if game.coups_sans_pomme > LIMITE_BOUCLE:
            return score, game.coups / serpent.GAME_SPEED, "boucle", game.coups
    fin = "victoire" if game.victoire else "mort"
    # Sans affichage, le temps de jeu se déduit des coups : un coup = 1/GAME_SPEED seconde
    temps = game.temps_ecoule() if game.affichage else game.coups / serpent.GAME_SPEED
    return score, temps, fin, game.coups


def evaluate(nom_algo, nb_parties=100, seed=None, args=None):
    """Mesure un algorithme sans affichage, comme en partie réelle."""
    if seed is not None:
        random.seed(seed)
    algo = construire_algo(args) if args else ALGORITHMES[nom_algo]()
    game = SnakeGameAlgo(affichage=False)
    resultats, explorees, coups_total, debut = [], 0, 0, time.time()

    for _ in range(nb_parties):
        score, temps, fin, coups = jouer_partie(game, algo)
        resultats.append((score, temps, fin))
        explorees += algo.cases_explorees
        coups_total += coups

    scores = [r[0] for r in resultats]
    total_temps = sum(r[1] for r in resultats)
    meilleur = max(resultats, key=lambda r: (r[0], -r[1]))  # meilleur score, puis le plus rapide
    print(f"Algorithme « {nom_algo} » sur {nb_parties} parties :")
    print(f"  Record          : {meilleur[0]} en {meilleur[1]:.1f}s de jeu")
    print(f"  Score moyen     : {sum(scores) / nb_parties:.2f} (médiane {sorted(scores)[nb_parties // 2]})")
    print(f"  Pommes/s de jeu : {sum(scores) / total_temps:.3f} (global)")
    print(f"  Cases explorées : {explorees / max(1, coups_total):.1f} par coup")
    print(f"  Calcul          : {1000 * (time.time() - debut) / max(1, coups_total):.2f} ms par coup")
    print(f"  Fins de partie  : " + ", ".join(f"{f} {sum(r[2] == f for r in resultats)}" for f in ("mort", "boucle", "victoire")))
    return resultats


def demo(nom_algo, facteur_vitesse=1, graine=None, args=None):
    """Partie réelle : affichage et clock d'origine, logs score et temps dans le terminal."""
    algo = construire_algo(args) if args else ALGORITHMES[nom_algo]()
    game = SnakeGameAlgo(affichage=True, facteur_vitesse=facteur_vitesse)
    font_fin = pygame.font.Font(None, 80)

    print("=" * 78)
    print("  SNAKE ALGO (groupe Fraise)")
    print(f"  Algorithme : {nom_algo}" + (f" | graine {graine} (partie reproductible)" if graine is not None else ""))
    print(f"  Grille {G}x{G} | Clock {serpent.GAME_SPEED} FPS | "
          "ESPACE = rejouer après une partie | Échap = quitter")
    if facteur_vitesse != 1:
        print(f"  MODE RAPIDE x{facteur_vitesse} : jeu à {serpent.GAME_SPEED * facteur_vitesse} FPS, "
              f"tous les temps affichés = temps réel x{facteur_vitesse}")
    print("=" * 78, flush=True)

    parties, record, numero = [], None, 0
    try:
        while True:
            numero += 1
            log(f"Partie {numero} | Début")
            score, temps, fin, _ = jouer_partie(game, algo, journal=log, numero=numero, graine=graine)
            fin_partie = {"victoire": "VICTOIRE (grille remplie)",
                          "mort": "GAME OVER (collision avec le corps)",
                          "boucle": "ARRÊT (boucle sans pomme)"}[fin]
            parties.append((numero, score, temps, fin))
            if record is None or (score, -temps) > (record[1], -record[2]):
                record = parties[-1]
            log(f"Partie {numero} | {fin_partie} | Score {score} | Temps {format_temps(temps)}")
            log(f"Record de la session : {record[1]} en {format_temps(record[2])} (partie {record[0]})"
                " | ESPACE = rejouer, Échap = quitter")

            # Écran de fin comme le jeu de base : score et chrono figés
            game._dessiner()
            if game.victoire:
                serpent.display_message(game.screen, font_fin, "VICTOIRE !", serpent.VERT)
            else:
                serpent.display_message(game.screen, font_fin, "GAME OVER", serpent.ROUGE)
            serpent.display_message(game.screen, game.font, "ESPACE pour rejouer.", serpent.BLANC, y_offset=100)
            pygame.display.flip()

            attente = True
            while attente:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        raise SystemExit
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        attente = False
                game.clock.tick(serpent.GAME_SPEED)
    except (SystemExit, KeyboardInterrupt):
        if not parties or parties[-1][0] != numero:
            log(f"Partie {numero} | Interrompue (non comptée)")
    finally:
        pygame.quit()

    print("=" * 78)
    print(f"  BILAN : {len(parties)} partie(s) terminée(s)")
    for n, s, t, fin in parties:
        print(f"    Partie {n} : score {s} en {format_temps(t)}" + (" VICTOIRE" if fin == "victoire" else ""))
    if record:
        print(f"  RECORD : {record[1]} en {format_temps(record[2])} (partie {record[0]})")
    print("=" * 78, flush=True)


# Par défaut : le glouton, plus rapide (victoire en ~13 min contre ~21 min pour l'hybride),
# au prix d'un taux de victoire d'environ 7 %. L'hybride reste disponible avec --algo hybride.
ALGO_PAR_DEFAUT = "glouton"

def construire_algo(args):
    """Crée l'algorithme demandé, avec les réglages passés en option."""
    if args.algo == "glouton":
        return Glouton(seuil_longueur=args.seuil, patience=args.patience, survie=args.survie)
    if args.algo == "audacieux":
        return Audacieux(controle=args.controle, survie=args.survie)
    return ALGORITHMES[args.algo]()


if __name__ == "__main__":
    # options communes, acceptées avant comme après le mode (demo / rapide / eval)
    communes = argparse.ArgumentParser(add_help=False)
    communes.add_argument("--algo", choices=list(ALGORITHMES), default=ALGO_PAR_DEFAUT)
    communes.add_argument("--graine", type=int, default=None,
                          help="rejoue exactement la même partie (même tirage des pommes)")
    communes.add_argument("--seuil", type=float, default=None, help="glouton : seuil de longueur")
    communes.add_argument("--patience", type=int, default=None,
                          help="glouton : coups sans pomme avant de retenter la pomme")
    communes.add_argument("--survie", default=None, choices=("espace", "mur", "loin_queue"),
                          help="façon de survivre quand on ne vise plus la pomme")
    communes.add_argument("--controle", default="queue", choices=("aucun", "espace", "queue"),
                          help="audacieux : niveau de vérification avant de foncer")

    parser = argparse.ArgumentParser(description="Snake algorithmique (groupe Fraise)", parents=[communes])
    # Sans argument : partie réelle avec le meilleur algorithme (commande du professeur)
    sous = parser.add_subparsers(dest="mode")
    sous.add_parser("demo", parents=[communes], help="(par défaut) partie réelle avec affichage et logs")
    p_rapide = sous.add_parser("rapide", parents=[communes],
                               help="partie accélérée, temps compté = temps réel x facteur")
    p_rapide.add_argument("facteur", type=int, nargs="?", default=10)
    p_eval = sous.add_parser("eval", parents=[communes], help="mesurer un algorithme sur N parties, sans affichage")
    p_eval.add_argument("--parties", type=int, default=100)
    p_eval.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.survie is None:
        args.survie = "loin_queue" if args.algo == "glouton" else "mur"

    if args.mode == "eval":
        evaluate(args.algo, nb_parties=args.parties, seed=args.seed, args=args)
    else:
        demo(args.algo, facteur_vitesse=args.facteur if args.mode == "rapide" else 1,
             graine=args.graine, args=args)
