"""Replay buffer prioritise (Prioritized Experience Replay, Schaul et al. 2015).

Version proportionnelle basee sur un SumTree : la probabilite de tirage d'une
transition est proportionnelle a sa priorite (|erreur TD| + eps) ** alpha.
Implementation 100% numpy, sans dependance a torch.

Optimisation : le SumTree expose des methodes vectorisees (`get_batch`,
`update_batch`) qui traitent tout un lot en O(log capacity) iterations
numpy au lieu d'une boucle Python de `batch_size` descentes/mises a jour.
Le buffer stocke aussi les transitions dans des tableaux numpy prealloues
(indexes par position circulaire) plutot que dans un tableau d'objets,
pour eviter les copies element par element lors de `sample()`.
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
        # Conserve pour compatibilite generique (SumTree utilise seul, hors PrioritizedReplayBuffer).
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

    def get_batch(self, values):
        """Version vectorisée de `get()` pour un lot de valeurs.

        Descend tout le lot simultanément niveau par niveau (au lieu de descendre
        chaque valeur séparément) : à chaque niveau, on compare la valeur restante
        au fils gauche pour tout le lot d'un coup avec numpy. Le nombre d'itérations
        est borné par la profondeur de l'arbre (~log2(capacity)), et non par la
        taille du lot. Les éléments ayant déjà atteint une feuille sont "gelés"
        (via np.where) pendant que les autres continuent de descendre — nécessaire
        car l'arbre n'est pas forcément parfaitement équilibré si `capacity` n'est
        pas une puissance de 2.

        Args:
            values: array-like (B,), valeurs à chercher dans [0, total).

        Returns:
            (tree_indices, priorities) : deux np.ndarray (B,), indices de feuilles
            (int64) et leurs priorités (float64).
        """
        values = np.array(values, dtype=np.float64, copy=True)
        idx = np.zeros(values.shape[0], dtype=np.int64)
        n_tree = len(self.tree)

        while True:
            left = 2 * idx + 1
            is_leaf = left >= n_tree
            if is_leaf.all():
                break
            # Clamp defensif des indices deja-feuilles pour rester dans les bornes
            # (leur valeur ne sera de toute facon pas utilisee, cf. np.where ci-dessous).
            left_c = np.minimum(left, n_tree - 1)
            right_c = np.minimum(left_c + 1, n_tree - 1)

            go_left = values <= self.tree[left_c]
            new_idx = np.where(go_left, left_c, right_c)
            new_values = np.where(go_left, values, values - self.tree[left_c])

            idx = np.where(is_leaf, idx, new_idx)
            values = np.where(is_leaf, values, new_values)

        return idx, self.tree[idx]

    def update_batch(self, tree_indices, priorities) -> None:
        """Version vectorisée de `update()` pour un lot d'indices/priorités.

        Écrit toutes les feuilles en une fois puis remonte vers la racine niveau
        par niveau, en recalculant chaque nœud comme la somme de ses deux enfants
        (plutôt que par delta) : cela reste correct même si un même indice de
        feuille apparaît plusieurs fois dans le lot (une transition tirée deux
        fois), auquel cas la dernière valeur du lot l'emporte. Les indices
        parents sont dédupliqués (`np.unique`) à chaque niveau pour éviter les
        écritures redondantes.

        Args:
            tree_indices: array-like (B,), indices de feuilles (comme rendus par `get`/`sample`).
            priorities: array-like (B,), nouvelles priorités correspondantes.
        """
        tree_indices = np.asarray(tree_indices, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float64)
        if tree_indices.size == 0:
            return

        # Doublons dans le lot : la derniere occurrence de chaque indice gagne.
        # (np.unique sur le tableau inverse => "premiere" occurrence inversee = derniere occurrence originale.)
        reversed_idx = tree_indices[::-1]
        _, first_pos_reversed = np.unique(reversed_idx, return_index=True)
        orig_pos = tree_indices.size - 1 - first_pos_reversed
        leaf_idx = tree_indices[orig_pos]
        leaf_pri = priorities[orig_pos]

        self.tree[leaf_idx] = leaf_pri

        if len(self.tree) == 1:
            return  # arbre a une seule feuille == racine, rien a propager

        current = np.unique((leaf_idx - 1) // 2)
        while True:
            left = 2 * current + 1
            right = 2 * current + 2
            self.tree[current] = self.tree[left] + self.tree[right]
            if current.size == 1 and current[0] == 0:
                break
            current = np.unique((current - 1) // 2)

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

    Les transitions sont stockées dans des tableaux numpy prealloues (states,
    actions, rewards, next_states, dones), indexes par position circulaire
    (identique a celle du SumTree sous-jacent). La dimension de l'etat est
    deduite paresseusement du premier `add()`, pour rester generique.
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

        # Stockage vectorise, alloue paresseusement au premier add() (dimension d'etat inconnue avant).
        self.state_dim = None
        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._dones = None

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

    def _allocate_storage(self, state_dim: int) -> None:
        """Alloue les tableaux numpy de stockage (appelé une seule fois, au 1er add)."""
        self.state_dim = state_dim
        self._states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.float32)

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
        state_arr = np.asarray(state, dtype=np.float32)
        next_state_arr = np.asarray(next_state, dtype=np.float32)

        if self._states is None:
            self._allocate_storage(state_arr.shape[0])

        pos = self.tree.write  # position d'ecriture circulaire (avant increment par tree.add)
        self._states[pos] = state_arr
        self._actions[pos] = int(action)
        self._rewards[pos] = float(reward)
        self._next_states[pos] = next_state_arr
        self._dones[pos] = float(done)

        # La donnee du SumTree n'est plus utilisee (stockage vectorise ci-dessus) :
        # on passe None pour eviter de dupliquer la transition dans un tableau d'objets.
        self.tree.add(self.max_priority, None)

    def sample(self, batch_size: int):
        """Échantillonne un lot de transitions par stratification prioritaire.

        Divise l'intervalle [0, total) en batch_size segments égaux et tire une
        valeur uniforme dans chacun. Cela réduit la variance par rapport à un
        tirage uniforme direct dans [0, total), tout en respectant la distribution
        de probabilité des priorités. La descente dans le SumTree et la lecture
        des transitions se font en une seule passe vectorisée (numpy) plutôt
        qu'en boucle Python.

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
        total = self.tree.total
        segment = total / batch_size

        offsets = np.arange(batch_size, dtype=np.float64)
        lows = segment * offsets
        highs = segment * (offsets + 1.0)
        values = np.random.uniform(lows, highs)
        # Clamp defensif : eviter de deborder sur total a cause des arrondis flottants.
        values = np.minimum(values, total - 1e-8)

        tree_indices, priorities = self.tree.get_batch(values)

        # Robustesse : priorite invalide (arrondi flottant) -> on retire uniformement dans [0, total).
        invalid = priorities <= 0.0
        while invalid.any():
            n_invalid = int(invalid.sum())
            retry_values = np.random.uniform(0.0, total - 1e-8, size=n_invalid)
            retry_idx, retry_pri = self.tree.get_batch(retry_values)
            tree_indices[invalid] = retry_idx
            priorities[invalid] = retry_pri
            invalid = priorities <= 0.0

        positions = tree_indices - (self.capacity - 1)
        states = self._states[positions]
        actions = self._actions[positions]
        rewards = self._rewards[positions]
        next_states = self._next_states[positions]
        dones = self._dones[positions]

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
        Mise à jour vectorisée dans le SumTree (cf. `SumTree.update_batch`).

        Args:
            tree_indices: array-like, indices retournés par sample().
            td_errors: array-like, erreurs TD calculées (cible - Q).
        """
        tree_indices = np.asarray(tree_indices, dtype=np.int64)
        td_errors = np.asarray(td_errors)
        new_priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        self.tree.update_batch(tree_indices, new_priorities)
        if new_priorities.size:
            self.max_priority = max(self.max_priority, float(new_priorities.max()))

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

    # 7) Coherence tree.total == somme des feuilles apres 500 update_batch avec doublons d'indices.
    rng = np.random.default_rng(42)
    tree7 = SumTree(capacity=1000)
    for i in range(1000):
        tree7.add(float(rng.uniform(0.1, 5.0)), i)
    leaf_start = tree7.capacity - 1
    for _ in range(500):
        # Indices tires dans une petite fenetre de 50 feuilles -> doublons quasi garantis sur 32 tirages.
        batch_idx = leaf_start + rng.integers(0, 50, size=32)
        batch_pri = rng.uniform(0.1, 5.0, size=32)
        tree7.update_batch(batch_idx, batch_pri)
    leaves_sum = float(tree7.tree[leaf_start:].sum())
    assert abs(tree7.total - leaves_sum) < 1e-6, (
        f"incoherence apres update_batch : total={tree7.total}, somme feuilles={leaves_sum}"
    )
    print(f"[7] OK tree.total == somme des feuilles apres 500 update_batch avec doublons (total={tree7.total:.4f})")

    # 8) Benchmark sample(128)+update_priorities(128) sur 50000 transitions : ancien vs nouveau.
    import importlib.util
    import time as _time

    _OLD_PATH = (
        r"C:\Users\louis\AppData\Local\Temp\claude\C--Users-louis-Documents-LeSerpent2026"
        r"\569511f5-14da-4cf0-a043-3efbd509decd\scratchpad\old_replay_buffer.py"
    )
    _spec = importlib.util.spec_from_file_location("old_replay_buffer", _OLD_PATH)
    _old_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_old_module)  # nom de module != "__main__" -> ses propres tests ne s'executent pas

    def _build_and_fill(cls, capacity=50_000, n_items=50_000, seed=0):
        b = cls(capacity=capacity, beta_frames=1_000_000)
        r = np.random.default_rng(seed)
        for _ in range(n_items):
            s = r.standard_normal(12).astype(np.float32)
            ns = r.standard_normal(12).astype(np.float32)
            a = int(r.integers(0, 3))
            rew = float(r.standard_normal())
            b.add(s, a, rew, ns, False)
        return b

    def _bench(buffer, n_iters=200, batch_size=128, seed=1):
        r = np.random.default_rng(seed)
        t0 = _time.perf_counter()
        for _ in range(n_iters):
            _, _, _, _, _, t_idx, _ = buffer.sample(batch_size)
            td_errors = r.standard_normal(batch_size).astype(np.float32)
            buffer.update_priorities(t_idx, td_errors)
        return _time.perf_counter() - t0

    old_buf = _build_and_fill(_old_module.PrioritizedReplayBuffer)
    new_buf = _build_and_fill(PrioritizedReplayBuffer)

    t_old = _bench(old_buf)
    t_new = _bench(new_buf)
    speedup = t_old / t_new if t_new > 0 else float("inf")
    print(
        f"[8] Benchmark sample(128)+update_priorities(128) x200 sur buffer de 50000 : "
        f"ancien={t_old:.3f}s, nouveau={t_new:.3f}s (x{speedup:.1f} plus rapide)"
    )
    assert t_new < t_old, "la nouvelle implementation devrait etre plus rapide que l'ancienne"

    print("=== Tous les tests sont passes ===")
