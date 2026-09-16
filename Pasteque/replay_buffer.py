"""Replay buffer prioritise (Prioritized Experience Replay, Schaul et al. 2015).

Version proportionnelle basee sur un SumTree : la probabilite de tirage d'une
transition est proportionnelle a sa priorite (|erreur TD| + eps) ** alpha.
Implementation 100% numpy, sans dependance a torch.
"""

from __future__ import annotations

import numpy as np


class SumTree:
    """Arbre binaire de sommes pour l'échantillonnage stratifié.

    Structure indexée permettant un échantillonnage efficace proportionnel aux
    priorités en temps O(log capacity). Les feuilles (les `capacity` dernières
    cases du tableau `tree`) contiennent les priorités des transitions ; chaque
    nœud interne contient la somme de ses deux enfants. Ainsi, la racine (indice 0)
    contient la somme totale des priorités. L'écriture des données se fait de
    manière circulaire (ring buffer).
    """

    def __init__(self, capacity: int):
        """Initialise un SumTree avec une capacité donnée.

        Args:
            capacity: int, nombre maximal de feuilles (transitions stockables).
        """
        self.capacity = int(capacity)
        # Arbre complet : capacity-1 noeuds internes + capacity feuilles.
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        # Donnees utilisateur associees a chaque feuille (memes indices que les feuilles, decales).
        self.data = np.empty(self.capacity, dtype=object)
        self.write = 0  # prochaine position d'ecriture (circulaire)
        self.count = 0  # nombre d'elements ecrits (sature a capacity)

    def add(self, priority: float, data) -> int:
        """Ajoute une donnée avec priorité, mode d'écriture circulaire.

        Args:
            priority: float, priorité initiale (p_i = (|TD| + eps) ** alpha).
            data: objet quelconque à stocker (transition tuple pour PER).

        Returns:
            int, indice dans le tableau `tree` de la feuille ajoutée.
        """
        tree_idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        return tree_idx

    def update(self, tree_idx: int, priority: float) -> None:
        """Met à jour la priorité d'une feuille et propage le changement à la racine.

        Opération O(log capacity). Utilisée lors du ré-échantillonnage prioritaire
        après calcul d'une nouvelle erreur TD.

        Args:
            tree_idx: int, indice dans le tableau `tree` de la feuille à mettre à jour.
            priority: float, nouvelle priorité.
        """
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def get(self, value: float):
        """Cherche la feuille dont la priorité cumulative contient `value`.

        Descente d'arbre O(log capacity) : utilisée pour l'échantillonnage stratifié.
        Divise le segment [0, total) en batch_size parties égales et tire une valeur
        uniforme dans chacune pour réduire la variance.

        Args:
            value: float, valeur à trouver dans [0, total).

        Returns:
            (tree_idx, priority, data) tuple : indice de feuille, priorité, et donnée.
        """
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                # Feuille atteinte.
                break
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right

        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self) -> float:
        """Retourne la somme totale des priorités (racine de l'arbre)."""
        return float(self.tree[0])

    def __len__(self) -> int:
        """Retourne le nombre de transitions actuellement stockées."""
        return self.count


class PrioritizedReplayBuffer:
    """Buffer de replay à priorités (Prioritized Experience Replay, PER).

    Sur-échantillonne les transitions à forte erreur TD (|TD| + eps) ** alpha,
    plutôt que les déplacements triviaux. Cela accélère l'apprentissage en
    concentrant les mises à jour sur les situations critiques (évitements de
    justesse, impasses potentielles). Les poids d'importance sampling (IS)
    corrigent le biais introduit par le sur-échantillonnage, avec un facteur
    de recuit beta croissant progressivement de beta_start vers 1.0 au fil de
    l'entraînement (pour passer progressivement d'un gradient biaisé à non-biaisé).
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        eps: float = 1e-6,
    ):
        """Initialise un buffer de replay à priorités.

        Args:
            capacity: int, nombre maximal de transitions à stocker (défaut 100k).
            alpha: float, exposant de priorité (0=uniform, 1=pure TD-error). Défaut 0.6.
            beta_start: float, facteur IS initial (défaut 0.4, croît vers 1.0).
            beta_frames: int, nombre d'appels sample() pour passer de beta_start à 1.0.
            eps: float, petite constante ajoutée aux erreurs TD pour stabilité (défaut 1e-6).
        """
        self.tree = SumTree(capacity)
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.frame = 0  # compteur d'appels a sample(), pour le recuit de beta
        self.max_priority = 1.0  # priorite max vue jusqu'ici

    def _beta(self) -> float:
        """Retourne le facteur d'importance sampling actuel (recuit linéaire).

        Beta croit de beta_start vers 1.0 progressivement sur beta_frames appels à
        sample(). Cela corrige graduellement le biais dû à la prioritisation : au
        début, on accepte un gradient biaisé (beta faible) pour explorer les traj.
        critiques ; à la fin, on utilise le gradient non-biaisé (beta=1.0).

        Returns:
            float, facteur IS pour l'époque courante.
        """
        progress = min(1.0, self.frame / float(self.beta_frames))
        return self.beta_start + progress * (1.0 - self.beta_start)

    def add(self, state, action, reward, next_state, done) -> None:
        """Ajoute une transition avec la priorité maximale observée.

        Les transitions nouvelles reçoivent max_priority pour garantir qu'elles
        seront rejouées au moins une fois avant de laisser l'erreur TD les
        déprioritariser. Cela évite de manquer les erreurs initialement hautes.

        Args:
            state: array-like (12,), observation courante.
            action: int ou action, action exécutée.
            reward: float, récompense reçue.
            next_state: array-like (12,), observation suivante.
            done: bool, fin de l'épisode.
        """
        transition = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        )
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size: int):
        """Échantillonne un lot de transitions par stratification prioritaire.

        Divise l'intervalle [0, total) en batch_size segments égaux et tire une
        valeur uniforme dans chacun. Cela réduit la variance par rapport à un
        tirage uniforme direct dans [0, total), tout en respectant la distribution
        de probabilité des priorités.

        Args:
            batch_size: int, nombre de transitions à retourner.

        Returns:
            tuple (states, actions, rewards, next_states, dones, tree_indices, is_weights) :
                - states: (batch_size, state_dim) float32
                - actions: (batch_size,) int64
                - rewards: (batch_size,) float32
                - next_states: (batch_size, state_dim) float32
                - dones: (batch_size,) float32
                - tree_indices: (batch_size,) int64, indices pour update_priorities()
                - is_weights: (batch_size,) float32, poids IS normalisés par le max.
        """
        n = len(self.tree)
        # Dimension de l'etat deduite de la premiere transition stockee (12 pour Snake).
        state_dim = len(self.tree.data[0][0])
        states = np.empty((batch_size, state_dim), dtype=np.float32)
        actions = np.empty(batch_size, dtype=np.int64)
        rewards = np.empty(batch_size, dtype=np.float32)
        next_states = np.empty((batch_size, state_dim), dtype=np.float32)
        dones = np.empty(batch_size, dtype=np.float32)
        tree_indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        total = self.tree.total
        segment = total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            # Clamp defensif : eviter de deborder sur total a cause des arrondis flottants.
            value = min(value, total - 1e-8)

            tree_idx, priority, data = self.tree.get(value)
            # Robustesse : priorite/donnee invalide (arrondi flottant) -> on retire pres de la racine.
            while data is None or priority <= 0.0:
                value = np.random.uniform(0, total - 1e-8)
                tree_idx, priority, data = self.tree.get(value)

            state, action, reward, next_state, done = data
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state
            dones[i] = done
            tree_indices[i] = tree_idx
            priorities[i] = priority

        # Poids d'importance sampling : (N * P(i))^(-beta), normalises par le max.
        beta = self._beta()
        probs = priorities / total
        is_weights = np.power(n * probs, -beta)
        is_weights /= is_weights.max()
        is_weights = is_weights.astype(np.float32)

        self.frame += 1
        return states, actions, rewards, next_states, dones, tree_indices, is_weights

    def update_priorities(self, tree_indices, td_errors) -> None:
        """Met à jour les priorités des transitions à partir de leurs erreurs TD.

        La priorité est calculée comme : p_i = (|TD_i| + eps) ** alpha
        où TD_i est l'erreur TD (cible - Q prédite). Les transitions avec
        un TD élevé (apprentissage imparfait) sont rehaussées en priorité,
        tandis que celles bien apprises (TD faible) sont déprioritarisées.

        Args:
            tree_indices: array-like, indices retournés par sample().
            td_errors: array-like, erreurs TD calculées (cible - Q).
        """
        tree_indices = np.asarray(tree_indices)
        td_errors = np.asarray(td_errors)
        new_priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for tree_idx, priority in zip(tree_indices, new_priorities):
            self.tree.update(int(tree_idx), float(priority))
            self.max_priority = max(self.max_priority, float(priority))

    def __len__(self) -> int:
        """Retourne le nombre de transitions actuellement stockées."""
        return len(self.tree)


if __name__ == "__main__":
    print("=== Tests replay_buffer.py ===")

    # 1) SumTree.total == somme des priorites ajoutees.
    tree = SumTree(capacity=4)
    priorities = [1.0, 2.0, 3.0, 4.0]
    for p in priorities:
        tree.add(p, f"data_{p}")
    assert abs(tree.total - sum(priorities)) < 1e-9, "total incorrect apres add"
    print(f"[1] OK total apres add = {tree.total} (attendu {sum(priorities)})")

    # 2) update() change correctement le total.
    idx0 = tree.capacity - 1  # feuille de la premiere case ecrite
    old_total = tree.total
    tree.update(idx0, 10.0)
    expected_total = old_total - priorities[0] + 10.0
    assert abs(tree.total - expected_total) < 1e-9, "total incorrect apres update"
    print(f"[2] OK total apres update = {tree.total} (attendu {expected_total})")

    # 3) Buffer de capacite 8 rempli avec 12 transitions -> len == 8.
    buf = PrioritizedReplayBuffer(capacity=8, beta_frames=100)
    for i in range(12):
        state = np.zeros(12, dtype=np.float32) + i
        buf.add(state, action=i % 3, reward=float(i), next_state=state, done=False)
    assert len(buf) == 8, f"len attendu 8, obtenu {len(buf)}"
    print(f"[3] OK len(buf) = {len(buf)} apres 12 add sur capacite 8")

    # 4) sample(4) -> bonnes shapes/dtypes, is_weights <= 1.
    states, actions, rewards, next_states, dones, tree_indices, is_weights = buf.sample(4)
    assert states.shape == (4, 12) and states.dtype == np.float32
    assert actions.shape == (4,) and actions.dtype == np.int64
    assert rewards.shape == (4,) and rewards.dtype == np.float32
    assert next_states.shape == (4, 12) and next_states.dtype == np.float32
    assert dones.shape == (4,) and dones.dtype == np.float32
    assert tree_indices.shape == (4,) and tree_indices.dtype == np.int64
    assert is_weights.shape == (4,) and is_weights.dtype == np.float32
    assert np.all(is_weights <= 1.0 + 1e-6)
    print("[4] OK shapes/dtypes de sample(4), is_weights <=1 :", is_weights)

    # 5) Une transition avec priorite 100x plus grande est tiree nettement plus souvent.
    buf2 = PrioritizedReplayBuffer(capacity=10, beta_frames=100)
    for i in range(10):
        state = np.zeros(12, dtype=np.float32) + i
        buf2.add(state, action=0, reward=0.0, next_state=state, done=False)
    # La transition d'indice 3 (dans l'ordre d'ajout) recoit une priorite 100x plus grande.
    target_tree_idx = 3 + buf2.tree.capacity - 1
    buf2.tree.update(target_tree_idx, 100.0)

    counts = {}
    n_draws = 2000
    for _ in range(n_draws // 1):  # tirages par lots de 1 pour compter chaque tree_idx individuellement
        _, _, _, _, _, tree_indices, _ = buf2.sample(1)
        idx = int(tree_indices[0])
        counts[idx] = counts.get(idx, 0) + 1
    target_count = counts.get(target_tree_idx, 0)
    other_counts = [c for idx, c in counts.items() if idx != target_tree_idx]
    avg_other = sum(other_counts) / len(other_counts) if other_counts else 0
    assert target_count > avg_other * 5, "la transition prioritaire n'est pas nettement plus tiree"
    print(f"[5] OK transition prioritaire tiree {target_count}/{n_draws} fois (moyenne autres = {avg_other:.1f})")

    # 6) beta evolue vers 1.0.
    buf3 = PrioritizedReplayBuffer(capacity=10, beta_start=0.4, beta_frames=10)
    for i in range(10):
        state = np.zeros(12, dtype=np.float32) + i
        buf3.add(state, action=0, reward=0.0, next_state=state, done=False)
    beta_initial = buf3._beta()
    for _ in range(50):
        buf3.sample(2)
    beta_final = buf3._beta()
    assert beta_initial < beta_final
    assert abs(beta_final - 1.0) < 1e-9
    print(f"[6] OK beta {beta_initial:.3f} -> {beta_final:.3f}")

    print("=== Tous les tests sont passes ===")
