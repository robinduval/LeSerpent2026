"""Bloc 'Model' : le reseau de neurones (Deep Q-Network) et son entrainement.

Deep Q-Learning (cf. PDF du cours, slide 3) : au lieu d'une table Q[state][action]
(impossible ici, l'espace d'etats est trop grand), un reseau de neurones apprend a
approximer la fonction Q(state, action) -> valeur attendue de recompense future.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Ancre le dossier de sauvegarde a l'emplacement de CE fichier (et non au
# repertoire courant du terminal, qui change selon d'ou on lance le script :
# c'etait la cause du "il a oublie sa progression" -- ./model pouvait pointer
# vers un dossier different a chaque lancement).
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


class Linear_QNet(nn.Module):
    """Petit MLP : 11 entrees (etat) -> couche cachee -> 4 sorties (une par action).

    Chaque sortie est la Q-value estimee de l'action correspondante pour l'etat donne.
    On choisit ensuite l'action de plus grande Q-value (cf. Agent.get_action).
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="model.pth"):
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(MODEL_DIR, file_name))

    def load(self, file_name="model.pth"):
        """Recharge des poids sauvegardes precedemment, si le fichier existe.
        Retourne True si un modele a bien ete recharge (sinon on repart
        d'un reseau initialise aleatoirement, comme avant)."""
        path = os.path.join(MODEL_DIR, file_name)
        if not os.path.exists(path):
            return False
        try:
            self.load_state_dict(torch.load(path))
            return True
        except RuntimeError:
            # Le fichier sauvegarde correspond a une architecture differente
            # (ex : on vient d'ajouter des entrees a l'etat) -> on repart
            # d'un reseau neuf plutot que de planter.
            print("Ancien modele incompatible (l'etat a change de taille) : reinitialisation.")
            return False


class QTrainer:
    """Implemente la mise a jour Q-learning (equation de Bellman) pour Linear_QNet.

    Q_new(s, a) = reward                                   si l'episode se termine ici
                = reward + gamma * max_a' Q(s', a')         sinon

    On entraine ensuite le reseau a rapprocher sa prediction Q(s, a) de cette cible
    (perte MSE), comme une regression classique.
    """

    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Reseau cible : une copie figee du modele, utilisee UNIQUEMENT pour
        # calculer max(Q(next_state)) dans la cible de Bellman. Sans ca, la
        # cible est calculee avec le meme reseau qu'on est en train de
        # modifier -> elle bouge a chaque mise a jour ("cible mouvante"),
        # ce qui ralentit et destabilise l'apprentissage, surtout avec des
        # recompenses tres etalees (0.1 a -10000+). On ne la resynchronise
        # qu'une fois de temps en temps (cf. update_target), pour qu'elle
        # reste stable le temps que le modele principal apprenne dessus.
        self.target_model = copy.deepcopy(model)

    def update_target(self):
        """Resynchronise le reseau cible sur le reseau principal. A appeler
        periodiquement (ex: toutes les N parties), pas a chaque pas."""
        self.target_model.load_state_dict(self.model.state_dict())

    @staticmethod
    def _compress(reward):
        """Compresse l'amplitude des recompenses avant de les utiliser dans le
        calcul de la cible : sign(r) * log(1 + |r|).

        Objectif : garder l'ORDRE (mourir reste pire que survivre, une grosse
        pomme reste meilleure qu'une petite) tout en evitant qu'un -10000
        ecrase numeriquement les nuances entre les recompenses de tous les
        jours (0.1 a quelques dizaines). Exemple : -10000 -> -9.2, +100 -> +4.6,
        +10 -> +2.4, +0.1 -> +0.1. C'est une simple fonction mathematique
        (pas un algorithme), appliquee seulement au calcul interne de la
        cible -- les recompenses "reelles" du jeu (celles que vous regardez,
        celles dans le PDF) ne changent pas.
        """
        return torch.sign(reward) * torch.log1p(torch.abs(reward))

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        # Permet de traiter un seul pas (short memory) ou un batch (long memory)
        # en ajoutant une dimension "batch" quand on ne recoit qu'un seul exemple.
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1. Q values predites pour l'etat actuel
        pred = self.model(state)

        # 2. Cible de Bellman : on ne corrige que la Q-value de l'action jouee,
        #    les autres restent inchangees (on ne penalise pas ce qui n'a pas ete choisi)
        compressed_reward = self._compress(reward)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = compressed_reward[idx]
            if not done[idx]:
                with torch.no_grad():
                    next_q = torch.max(self.target_model(next_state[idx]))
                Q_new = compressed_reward[idx] + self.gamma * next_q

            action_idx = torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q_new

        # 3. Backpropagation classique (MSE entre prediction et cible)
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # Le -10000 sur la mort cree un ecart enorme avec les autres
        # recompenses (dizaines/centaines) : la premiere fois que l'agent
        # meurt, l'erreur (et donc le gradient) est gigantesque et peut
        # "casser" les poids du reseau (mise a jour disproportionnee).
        # Le clipping rogne juste la NORME du vecteur gradient a 10 si elle
        # la depasse (garde la direction, reduit l'amplitude) : simple
        # operation mathematique, pas un algorithme de plus.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)

        self.optimizer.step()
