"""
snake-ia-gpu.py — entraîneur vectorisé pour `snake-ia.py`. Groupe ABRICOT.

CE N'EST PAS LE LIVRABLE, et — RÉSULTAT MESURÉ — CE N'EST PAS NON PLUS UN
ACCÉLÉRATEUR UTILE POUR CE PROBLÈME. Le fichier est conservé parce que la
mesure qui l'invalide est instructive, et parce qu'il redeviendrait pertinent
sur un réseau plus gros.

CE QU'ON VOULAIT FAIRE
----------------------
Le profilage de `snake-ia.py` donne ~763 us pour un `train_step` sur UN
échantillon, réseau 16->256->3, et 84 % du temps total passé dans PyTorch. Ce
n'est pas du calcul mais de la surcharge de framework : chaque appel paie un
coût fixe quelle que soit la taille du tenseur. Un GPU ne réduit pas ce coût,
il l'aggrave (latence de lancement de noyau, transferts hôte<->carte).

La seule façon de rentabiliser un accélérateur est de grossir les tenseurs.
D'où ce fichier : N parties menées de front (tenseurs empilés), une passe avant
servant N parties d'un coup.

CE QUE ÇA DONNE RÉELLEMENT
--------------------------
Débit de simulation : x15 (0,52 -> 7,8 parties/s sur le même CPU). Mais à TEMPS
ÉGAL, sur la même machine, le séquentiel gagne largement :

    à 386 s   séquentiel : 228 parties, moy50 87,96
              vectorisé : 3000 parties, moy50 46,64

POURQUOI — et c'est le point à retenir. Le séquentiel fait ~1 mise à jour de
gradient par pas de jeu. Le vectorisé en fait 1 pour N pas produits : le
rapport gradient/expérience tombe à 1/N. Il génère l'expérience N fois plus
vite et l'exploite N fois moins.

Compenser par `--maj N` rétablit le rapport, mais ramène mécaniquement le même
coût total de mises à jour : le gain se réduit alors à la part simulation, soit
16 % du temps. PLAFOND THÉORIQUE DE L'ACCÉLÉRATION : 1 / 0,84 = ~1,19x.
Vérifié empiriquement — à 300 s et 64 environnements, `--maj 1` et `--maj 8`
plafonnent tous deux à moy50 ~1.

CONCLUSION : le goulot n'a jamais été la génération d'expérience, mais les mises
à jour de gradient — et celles-ci ne se parallélisent pas entre environnements,
puisqu'il en faut d'autant plus qu'on produit plus d'expérience. La
vectorisation paierait sur un réseau assez gros pour que le calcul domine la
surcharge de framework ; sur un 16->256->3, elle ne peut pas.

CE QUI RESTE VALABLE ICI
------------------------
Le flood-fill reformulé en DILATATION ITÉRATIVE de masque : on décale le masque
« atteint » des quatre côtés (`torch.roll`, qui sur un tore EST le passage par
les bords), on intersecte avec les cases libres, on répète jusqu'à
stabilisation. Un parcours en largeur irrégulier devient du calcul tensoriel
régulier, applicable à N parties simultanément.

PARITÉ — les deux garanties vérifiées
-------------------------------------
    python snake-ia-gpu.py --parite     # 16 features vs snake-ia.py

État : écart max 2,55e-08 sur 200 états tirés au hasard (arrondi flottant).
Dynamique : 30/30 parties strictement identiques à la référence pour une même
suite d'actions. Ces deux tests ont chacun révélé un vrai bug — le flood-fill
partant de la tête (qui occupe sa propre case, d'où `depart_libre`), et les
pointeurs de tête et de queue du tampon circulaire qui partaient en sens
opposés. Un entraîneur rapide mais faux n'aurait produit que des poids
silencieusement inutilisables.

Usage :
    python snake-ia-gpu.py --parties 4000 --envs 256 --maj 8
    python snake-ia.py bench --bareme potentiel --etat conscient --securite \
           --model model/snake-ia-gpu.pth
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- constantes du jeu, identiques à snake-ia.py -------------------------
GRID = 15
CASES = GRID * GRID
# Ordre horaire : RIGHT, DOWN, LEFT, UP. Indispensable pour que « tourner à
# droite » soit +1 et « tourner à gauche » -1, comme dans le livrable.
DELTA = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=torch.long)

# --- barème « potentiel » de snake-ia.py ---------------------------------
R_POMME, R_MORT, R_VICTOIRE, R_PAS = 10.0, -10.0, 100.0, -0.01
PHI_POMME, PHI_ESPACE = 2.0, 1.0
GAMMA, GAMMA_FACONNAGE = 0.9, 1.0

LR, CACHE = 0.001, 256


def choisir_device(demande):
    if demande != "auto":
        return torch.device(demande)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


class JeuxParalleles:
    """N parties de Snake menées de front, entièrement en tenseurs.

    Le corps est un tampon circulaire (N, CASES, 2) : pousser une tête écrit à
    `tete_ptr`, retirer la queue avance `queue_ptr`. Aucune liste Python, donc
    aucune boucle sur les parties.
    """

    def __init__(self, n, device):
        self.n, self.device = n, device
        self.delta = DELTA.to(device)
        self.ar = torch.arange(n, device=device)
        self.reset_tout()

    # ------------------------------------------------------------------
    def reset_tout(self):
        n, dev = self.n, self.device
        self.occ = torch.zeros(n, GRID, GRID, dtype=torch.bool, device=dev)
        self.corps = torch.zeros(n, CASES, 2, dtype=torch.long, device=dev)
        self.tete_ptr = torch.zeros(n, dtype=torch.long, device=dev)
        self.queue_ptr = torch.zeros(n, dtype=torch.long, device=dev)
        self.longueur = torch.full((n,), 3, dtype=torch.long, device=dev)
        self.dir = torch.zeros(n, dtype=torch.long, device=dev)  # RIGHT
        self.score = torch.zeros(n, dtype=torch.long, device=dev)
        self.pas = torch.zeros(n, dtype=torch.long, device=dev)
        self.frame = torch.zeros(n, dtype=torch.long, device=dev)
        self.phi = None

        # même position de départ que le livrable : tête en (GRID//4, GRID//2),
        # corps étendu vers la gauche
        hx, hy = GRID // 4, GRID // 2
        for k, dx in enumerate((0, -1, -2)):
            self.corps[:, k, 0] = hx + dx
            self.corps[:, k, 1] = hy
            self.occ[:, hy, (hx + dx) % GRID] = True
        self.tete_ptr[:] = 0
        self.queue_ptr[:] = 2
        self.tete = self.corps[:, 0].clone()
        self._poser_pomme(torch.ones(self.n, dtype=torch.bool, device=dev))
        return self

    def reset_masque(self, m):
        """Réinitialise seulement les parties marquées par `m`."""
        if not bool(m.any()):
            return
        idx = self.ar[m]
        self.occ[idx] = False
        self.corps[idx] = 0
        hx, hy = GRID // 4, GRID // 2
        for k, dx in enumerate((0, -1, -2)):
            self.corps[idx, k, 0] = hx + dx
            self.corps[idx, k, 1] = hy
            self.occ[idx, hy, (hx + dx) % GRID] = True
        self.tete_ptr[idx] = 0
        self.queue_ptr[idx] = 2
        self.longueur[idx] = 3
        self.dir[idx] = 0
        self.score[idx] = 0
        self.pas[idx] = 0
        self.frame[idx] = 0
        self.tete[idx] = self.corps[idx, 0]
        self._poser_pomme(m)

    # ------------------------------------------------------------------
    def _poser_pomme(self, m):
        """Pose une pomme sur une case libre, pour les parties marquées.

        Tirage vectorisé : on donne un score aléatoire aux cases libres, -inf
        aux occupées, et on prend l'argmax. Équivalent à un choix uniforme
        parmi les libres, sans boucle.
        """
        if not bool(m.any()):
            return
        bruit = torch.rand(self.n, CASES, device=self.device)
        bruit = bruit.masked_fill(self.occ.view(self.n, CASES), -1.0)
        plat = bruit.argmax(dim=1)
        pomme = torch.stack([plat % GRID, plat // GRID], dim=1)
        if not hasattr(self, "pomme"):
            self.pomme = pomme
        else:
            self.pomme = torch.where(m.unsqueeze(1), pomme, self.pomme)
        # plus aucune case libre -> victoire (traitée par l'appelant)
        self.plus_de_place = (~self.occ.view(self.n, CASES)).sum(1) == 0

    def _case_queue(self):
        return self.corps[self.ar, self.queue_ptr]

    # ------------------------------------------------------------------
    def flood(self, depart, libres, cap=None, depart_libre=False):
        """Flood-fill vectorisé, par dilatation itérative de masque.

        `depart` : (N,2) case de départ. `libres` : (N,GRID,GRID) cases
        traversables. Rend (nb atteignable, masque atteint).

        Un `roll` sur un tore EST le passage par les bords : c'est exactement
        la topologie du jeu, donc aucune correction de bord n'est nécessaire.
        """
        atteint = torch.zeros(self.n, GRID, GRID, dtype=torch.bool,
                              device=self.device)
        # `depart_libre` : on part de la TÊTE, qui occupe sa propre case. Sans
        # ce drapeau le masque la juge bloquée et le parcours rend 0 — c'est
        # exactement le bug qui rendait muettes `queue_ok` et `espace_total`
        # dans le prototype. Le livrable a le même drapeau, même raison.
        if depart_libre:
            atteint[self.ar, depart[:, 1], depart[:, 0]] = True
        else:
            atteint[self.ar, depart[:, 1], depart[:, 0]] = \
                libres[self.ar, depart[:, 1], depart[:, 0]]

        # Diamètre du tore = 14 ; avec obstacles un chemin peut serpenter, d'où
        # la borne CASES. On sort dès stabilisation, ce qui arrive bien avant.
        for _ in range(CASES):
            voisins = (torch.roll(atteint, 1, dims=2)
                       | torch.roll(atteint, -1, dims=2)
                       | torch.roll(atteint, 1, dims=1)
                       | torch.roll(atteint, -1, dims=1))
            neuf = (voisins & libres) | atteint
            if bool((neuf == atteint).all()):
                break
            atteint = neuf
        if cap is None:
            return atteint.view(self.n, CASES).sum(1), atteint
        n = atteint.view(self.n, CASES).sum(1)
        return torch.clamp(n, max=cap), atteint

    def _libres_hors_queue(self):
        """Cases traversables : les vides, PLUS la case de queue.

        Même règle que `is_collision` du livrable : la queue va se libérer ce
        tour-ci, donc elle ne bloque pas.
        """
        libres = ~self.occ
        q = self._case_queue()
        libres[self.ar, q[:, 1], q[:, 0]] = True
        return libres

    def collision(self, cases):
        """(N,2) -> booléen : la case tue-t-elle ? Queue exclue."""
        c = cases % GRID
        occupe = self.occ[self.ar, c[:, 1], c[:, 0]]
        q = self._case_queue()
        est_queue = (c[:, 0] == q[:, 0]) & (c[:, 1] == q[:, 1])
        return occupe & ~est_queue

    # ------------------------------------------------------------------
    def offset(self, a, b):
        """Déplacement signé le plus court de a vers b, sur le tore."""
        d = (b - a) % GRID
        return torch.where(d > GRID // 2, d - GRID, d)

    def distance(self, a, b):
        o = self.offset(a, b).abs()
        return o[:, 0] + o[:, 1]

    def potentiel(self):
        prox = 1.0 - self.distance(self.tete, self.pomme).float() / (GRID - 1)
        espace, _ = self.flood(self.tete, self._libres_hors_queue(),
                               depart_libre=True)
        return PHI_POMME * prox + PHI_ESPACE * espace.float() / CASES

    # ------------------------------------------------------------------
    def etat(self):
        """Les 16 features « conscient », dans l'ordre exact du livrable."""
        i = self.dir
        d_s = self.delta[i]
        d_r = self.delta[(i + 1) % 4]
        d_l = self.delta[(i - 1) % 4]
        c_s = (self.tete + d_s) % GRID
        c_r = (self.tete + d_r) % GRID
        c_l = (self.tete + d_l) % GRID

        libres = self._libres_hors_queue()
        cap = self.longueur + 1
        esp = []
        for c in (c_s, c_r, c_l):
            n, _ = self.flood(c, libres)
            # le livrable rend 0 si la case de départ est bloquée
            n = torch.where(self.collision(c), torch.zeros_like(n), n)
            esp.append(torch.clamp(n.float() / cap.float(), max=1.0))

        libre, atteint = self.flood(self.tete, libres, depart_libre=True)
        q = self._case_queue()
        queue_ok = atteint[self.ar, q[:, 1], q[:, 0]].float()

        off = self.offset(self.tete, self.pomme).float()
        demi = GRID // 2
        devant = (off[:, 0] * d_s[:, 0] + off[:, 1] * d_s[:, 1]) / demi
        cote = (off[:, 0] * d_r[:, 0] + off[:, 1] * d_r[:, 1]) / demi
        dnorm = self.distance(self.tete, self.pomme).float() / (GRID - 1)

        return torch.stack([
            self.collision(c_s).float(),
            self.collision(c_r).float(),
            self.collision(c_l).float(),
            esp[0], esp[1], esp[2],
            (i == 2).float(), (i == 0).float(),
            (i == 3).float(), (i == 1).float(),
            devant, cote, dnorm,
            self.longueur.float() / CASES,
            queue_ok,
            libre.float() / CASES,
        ], dim=1)

    # ------------------------------------------------------------------
    def avancer(self, action):
        """Un pas pour les N parties. action : (N,) dans {0,1,2}."""
        self.pas += 1
        self.frame += 1
        if self.phi is None:
            self.phi = self.potentiel()
        phi_avant = self.phi

        virage = torch.tensor([0, 1, -1], device=self.device)[action]
        self.dir = (self.dir + virage) % 4
        neuf = (self.tete + self.delta[self.dir]) % GRID

        mort = self.collision(neuf) | (self.frame > 100 * self.longueur)
        vivant = ~mort

        grandit = ((neuf[:, 0] == self.pomme[:, 0])
                   & (neuf[:, 1] == self.pomme[:, 1]) & vivant)
        retire = vivant & ~grandit

        # queue retirée AVANT l'ajout de la tête : si la tête vient occuper
        # l'ancienne case de queue, l'ordre inverse effacerait la tête
        if bool(retire.any()):
            idx = self.ar[retire]
            q = self.corps[idx, self.queue_ptr[idx]]
            self.occ[idx, q[:, 1], q[:, 0]] = False
            # MÊME SENS que tete_ptr, qui recule. L'invariant du tampon est
            # queue_ptr == tete_ptr + longueur - 1 : si la tête recule d'un cran
            # et que la longueur ne change pas, la queue doit reculer aussi.
            # Les faire diverger fait pointer la queue sur de la mémoire vide
            # dès le deuxième pas, et le serpent meurt aussitôt.
            self.queue_ptr[idx] = (self.queue_ptr[idx] - 1) % CASES

        if bool(vivant.any()):
            idx = self.ar[vivant]
            self.tete_ptr[idx] = (self.tete_ptr[idx] - 1) % CASES
            self.corps[idx, self.tete_ptr[idx]] = neuf[idx]
            self.occ[idx, neuf[idx][:, 1], neuf[idx][:, 0]] = True
            self.tete[idx] = neuf[idx]
        self.longueur = self.longueur + grandit.long()
        self.score = self.score + grandit.long()
        self.frame = torch.where(grandit, torch.zeros_like(self.frame),
                                 self.frame)

        self._poser_pomme(grandit)
        victoire = grandit & self.plus_de_place
        fini = mort | victoire

        recompense = torch.full((self.n,), R_PAS, device=self.device)
        recompense = torch.where(grandit, torch.full_like(recompense, R_POMME),
                                 recompense)
        recompense = torch.where(mort, torch.full_like(recompense, R_MORT),
                                 recompense)
        recompense = torch.where(victoire,
                                 torch.full_like(recompense, R_VICTOIRE),
                                 recompense)

        self.phi = self.potentiel()
        faconnage = GAMMA_FACONNAGE * self.phi - phi_avant
        recompense = recompense + torch.where(mort | grandit | victoire,
                                              torch.zeros_like(faconnage),
                                              faconnage)
        return recompense, fini


class Linear_QNet(nn.Module):
    """Même architecture que le livrable : les poids sont interchangeables."""

    def __init__(self, n_in=16, n_h=CACHE, n_out=3):
        super().__init__()
        self.linear1 = nn.Linear(n_in, n_h)
        self.linear2 = nn.Linear(n_h, n_out)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


def entrainer(parties, n_env, device, sortie, memoire=200_000, lot=4096,
              maj=1):
    dev = choisir_device(device)
    print(f"device : {dev}  |  {n_env} parties en parallèle  |  cible {parties} parties")
    jeux = JeuxParalleles(n_env, dev)
    net = Linear_QNet().to(dev)
    opt = optim.Adam(net.parameters(), lr=LR)
    perte = nn.MSELoss()

    S = torch.zeros(memoire, 16, device=dev)
    A = torch.zeros(memoire, dtype=torch.long, device=dev)
    R = torch.zeros(memoire, device=dev)
    S2 = torch.zeros(memoire, 16, device=dev)
    D = torch.zeros(memoire, dtype=torch.bool, device=dev)
    curseur, remplis = 0, 0

    finies, scores, record = 0, [], 0
    prochain_log = 50
    os.makedirs("model", exist_ok=True)
    log = csv.writer(open(sortie.replace(".pth", ".csv"), "w", newline=""))
    log.writerow(["parties", "score_moyen_50", "record", "secondes"])
    t0 = time.time()

    while finies < parties:
        etat = jeux.etat()
        eps = max(0.02, 1.0 - finies / (0.6 * parties))
        with torch.no_grad():
            q = net(etat)
        act = q.argmax(1)
        hasard = torch.rand(n_env, device=dev) < eps
        act = torch.where(hasard, torch.randint(0, 3, (n_env,), device=dev), act)

        rec, fini = jeux.avancer(act)
        etat2 = jeux.etat()

        k = act.shape[0]
        pos = (torch.arange(k, device=dev) + curseur) % memoire
        S[pos], A[pos], R[pos], S2[pos], D[pos] = etat, act, rec, etat2, fini
        curseur = int((curseur + k) % memoire)
        remplis = min(remplis + k, memoire)

        if bool(fini.any()):
            sc = jeux.score[fini].tolist()
            scores.extend(sc)
            finies += len(sc)
            record = max([record] + sc)
            jeux.reset_masque(fini)
            jeux.phi = jeux.potentiel()

        # `maj` mises à jour par pas d'environnement. Avec N environnements,
        # un pas produit N transitions mais ne déclenche qu'une mise à jour :
        # le rapport « gradient / expérience » tombe à 1/N, alors que la
        # version séquentielle tourne à ~1. C'est ce déséquilibre, et non la
        # vitesse de simulation, qui décide de l'apprentissage par seconde.
        if remplis >= lot:
            for _ in range(maj):
                ech = torch.randint(0, remplis, (lot,), device=dev)
                pred = net(S[ech])
                q_next = net(S2[ech]).max(1).values
                cible = pred.clone()
                cible[torch.arange(lot, device=dev), A[ech]] = (
                    R[ech] + GAMMA * q_next * (~D[ech]).float())
                opt.zero_grad()
                perte(cible, pred).backward()
                opt.step()

        if finies >= prochain_log and scores:
            prochain_log = finies + 50
            m = float(np.mean(scores[-50:]))
            el = time.time() - t0
            print(f"  {finies:5d} parties | moy50 {m:6.2f} | record {record:3d} "
                  f"| {el:6.1f}s | {finies/el:5.1f} parties/s", flush=True)
            log.writerow([finies, round(m, 2), record, round(el, 1)])
            torch.save(net.state_dict(), sortie)

    torch.save(net.state_dict(), sortie)
    el = time.time() - t0
    print(f"\nterminé : {finies} parties en {el:.0f}s ({finies/el:.1f} parties/s)")
    print(f"record {record} | moyenne des 50 dernières {np.mean(scores[-50:]):.2f}")
    print(f"modèle -> {sortie}  (rechargeable par snake-ia.py)")


def test_parite(n=200):
    """Compare les 16 features avec celles de snake-ia.py, état par état.

    Sans cette garantie, les poids appris ici seraient inexploitables par le
    livrable : mêmes dimensions, mais un sens différent par colonne.
    """
    import importlib.util
    import random
    spec = importlib.util.spec_from_file_location("ia", "snake-ia.py")
    ia = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ia)

    dev = torch.device("cpu")
    pire, teste = 0.0, 0
    for essai in range(n):
        random.seed(essai)
        ref = ia.SnakeGameAI(scheme="potentiel")
        ref.reset()
        # on fait vivre la partie pour tomber sur des serpents longs
        for _ in range(random.randint(0, 250)):
            a = [0, 0, 0]
            a[random.randint(0, 2)] = 1
            if ref.play_step(a)[1]:
                ref.reset()

        vec = JeuxParalleles(1, dev)
        # on recopie l'état de référence dans la version vectorisée
        corps = list(ref.body)
        vec.occ[:] = False
        vec.corps[:] = 0
        for k, (x, y) in enumerate(corps):
            vec.corps[0, k, 0], vec.corps[0, k, 1] = x, y
            vec.occ[0, y, x] = True
        vec.tete_ptr[0], vec.queue_ptr[0] = 0, len(corps) - 1
        vec.longueur[0] = len(corps)
        vec.tete[0, 0], vec.tete[0, 1] = ref.head
        vec.dir[0] = ia.CLOCKWISE.index(ref.direction)
        vec.pomme[0, 0], vec.pomme[0, 1] = ref.food

        attendu = ia.Agent.build_state(ref, "conscient")
        obtenu = vec.etat()[0].numpy()
        ecart = float(np.abs(attendu - obtenu).max())
        pire = max(pire, ecart)
        teste += 1

    noms = ["dgr_face", "dgr_dr", "dgr_ga", "esp_face", "esp_dr", "esp_ga",
            "dir_L", "dir_R", "dir_U", "dir_D", "pom_devant", "pom_cote",
            "pom_dist", "longueur", "queue_ok", "espace_tot"]
    print(f"parité vérifiée sur {teste} états tirés au hasard")
    print(f"écart maximal sur les 16 features : {pire:.3e}")
    if pire < 1e-5:
        print("PARITÉ OK — les poids entraînés ici sont valides pour snake-ia.py")
        return True
    print("PARITÉ ROMPUE — ne pas utiliser les poids")
    print("colonnes :", ", ".join(f"{i}:{n}" for i, n in enumerate(noms)))
    return False


def main():
    p = argparse.ArgumentParser(
        description="Entraîneur vectorisé (GPU-ready) pour snake-ia.py.")
    p.add_argument("--parties", type=int, default=4000)
    p.add_argument("--envs", type=int, default=256,
                   help="parties menées en parallèle (gros = GPU rentable)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "xpu"])
    p.add_argument("--sortie", default="model/snake-ia-gpu.pth")
    p.add_argument("--maj", type=int, default=8,
                   help="mises à jour de gradient par pas d'environnement")
    p.add_argument("--parite", action="store_true",
                   help="vérifie la parité des features avec snake-ia.py")
    a = p.parse_args()
    if a.parite:
        raise SystemExit(0 if test_parite() else 1)
    entrainer(a.parties, a.envs, a.device, a.sortie, maj=a.maj)


if __name__ == "__main__":
    main()
