"""Modèle de réseau de neurones Dueling DQN pour l'agent Snake RL.

Implémente une architecture Dueling double tête (value et advantage streams)
qui sépare l'estimation de la valeur V(s) de celle de l'avantage A(s,a).
Cette séparation est pertinente pour Snake car beaucoup d'états (longs couloirs
libres) sont "neutres" (l'action importe peu), tandis que les impasses exigent
une sélection d'action précise.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """Reseau Dueling DQN : separe l'estimation de V(s) et de A(s,a).

    On separe la valeur d'un etat V(s) de l'avantage de chaque action A(s,a)
    car dans beaucoup d'etats (ex: long couloir vide) l'action choisie importe
    peu, alors que dans d'autres (ex: impasse pres d'un mur) le choix precis
    de l'action est crucial pour la survie du serpent.
    """

    def __init__(self, input_size=12, hidden_size=256, output_size=4):
        """Initialise le réseau Dueling DQN avec tronc commun et deux têtes.

        Args:
            input_size: int, dimension d'entrée (6 rayons × 2 par raycast = 12 pour Snake).
            hidden_size: int, nombre de neurones dans le tronc commun (défaut 256).
            output_size: int, nombre d'actions (4 pour Snake : UP/DOWN/LEFT/RIGHT).
        """
        super().__init__()
        # Tronc commun partage par les deux tetes.
        self.tronc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Tete valeur : estime V(s), un seul scalaire par etat.
        self.tete_valeur = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Tete avantage : estime A(s,a), un score par action.
        self.tete_avantage = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward_components(self, x):
        """Calcule séparément V(s) et A(s,a) après le tronc commun.

        Utile pour inspecter les deux composantes du réseau ou implémenter
        des variantes d'agrégation.

        Args:
            x: tensor de shape (batch, input_size) ou (input_size,).

        Returns:
            (V, A) tuple : V de shape (batch, 1), A de shape (batch, output_size).
        """
        est_vecteur_seul = x.dim() == 1
        if est_vecteur_seul:
            x = x.unsqueeze(0)

        caracteristiques = self.tronc(x)
        valeur = self.tete_valeur(caracteristiques)
        avantage = self.tete_avantage(caracteristiques)
        return valeur, avantage

    def forward(self, x):
        """Combine V(s) et A(s,a) pour obtenir les valeurs Q(s,a).

        Agrégation Dueling classique : Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

        La soustraction de la moyenne des avantages assure l'identifiabilité
        du modèle : elle rend V et A uniquement déterminés. Sans elle, on
        pourrait ajouter une constante à V et la soustraire de tous les A sans
        changer Q, ce qui rendrait l'apprentissage instable.

        Args:
            x: tensor de shape (batch, input_size) ou (input_size,).

        Returns:
            tensor Q de shape (batch, output_size) ou (output_size,).
        """
        est_vecteur_seul = x.dim() == 1

        valeur, avantage = self.forward_components(x)
        moyenne_avantage = avantage.mean(dim=1, keepdim=True)
        q_valeurs = valeur + (avantage - moyenne_avantage)

        if est_vecteur_seul:
            q_valeurs = q_valeurs.squeeze(0)
        return q_valeurs

    def save(self, file_path):
        """Sauvegarde les poids du modèle sur disque.

        Crée les répertoires parents si nécessaire. Les poids sont enregistrés
        dans le format PyTorch standard (dict d'état).

        Args:
            file_path: str, chemin du fichier de sauvegarde (ex: "checkpoints/model.pth").
        """
        dossier = os.path.dirname(file_path)
        if dossier and not os.path.exists(dossier):
            os.makedirs(dossier)
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        """Charge les poids du modèle depuis le disque et passe en mode évaluation.

        Place le réseau en mode eval() (batch norm et dropout désactivés) pour
        assurer des prédictions déterministes lors de l'inférence.

        Args:
            file_path: str, chemin du fichier de poids à charger.

        Returns:
            self (DuelingDQN), pour chaînage d'appels.
        """
        etat = torch.load(file_path, map_location="cpu")
        self.load_state_dict(etat)
        self.eval()
        return self


if __name__ == "__main__":
    import tempfile

    modele = DuelingDQN(input_size=12, hidden_size=256, output_size=4)

    # --- Test 1 : shape de sortie avec un batch ---
    entree_batch = torch.randn(8, 12)
    sortie_batch = modele(entree_batch)
    assert sortie_batch.shape == (8, 4), f"Shape batch incorrecte : {sortie_batch.shape}"
    print(f"Test batch OK : shape sortie = {tuple(sortie_batch.shape)}")

    # --- Test 2 : shape de sortie avec un seul vecteur (12,) ---
    entree_vecteur = torch.randn(12)
    sortie_vecteur = modele(entree_vecteur)
    assert sortie_vecteur.shape == (4,), f"Shape vecteur incorrecte : {sortie_vecteur.shape}"
    print(f"Test vecteur seul OK : shape sortie = {tuple(sortie_vecteur.shape)}")

    # --- Test 3 : invariant dueling -> moyenne de (Q - V) nulle sur les actions ---
    valeur, avantage = modele.forward_components(entree_batch)
    q_valeurs = modele(entree_batch)
    difference = q_valeurs - valeur  # broadcast (8,4) - (8,1)
    moyenne_par_etat = difference.mean(dim=1)
    assert torch.allclose(moyenne_par_etat, torch.zeros_like(moyenne_par_etat), atol=1e-6), (
        f"L'invariant dueling n'est pas respecte : {moyenne_par_etat}"
    )
    print(f"Test invariant dueling OK : moyenne(Q - V) par etat = {moyenne_par_etat.detach().numpy()}")

    # --- Test 4 : sauvegarde / chargement dans un fichier temporaire ---
    with tempfile.TemporaryDirectory() as dossier_temp:
        chemin_temp = os.path.join(dossier_temp, "sous_dossier", "modele_test.pth")
        modele.save(chemin_temp)
        assert os.path.exists(chemin_temp), "Le fichier de sauvegarde n'a pas ete cree."

        nouveau_modele = DuelingDQN(input_size=12, hidden_size=256, output_size=4)
        nouveau_modele.load(chemin_temp)

        with torch.no_grad():
            sortie_originale = modele(entree_batch)
            sortie_chargee = nouveau_modele(entree_batch)
        assert torch.allclose(sortie_originale, sortie_chargee, atol=1e-6), (
            "Les sorties different apres chargement du modele."
        )
        assert not nouveau_modele.training, "Le modele charge devrait etre en mode eval()."
        print("Test save/load OK : sorties identiques apres round-trip, modele en eval().")
    # Le fichier temporaire est supprime automatiquement a la sortie du bloc 'with'.
    print("Fichier temporaire supprime.")

    print("Tous les tests sont passes avec succes.")
