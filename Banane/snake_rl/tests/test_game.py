"""Tests du moteur corrigé : collisions, score, croissance, victoire."""

import pytest

from snake_rl import rules
from snake_rl.game import COURSE_REWARDS, RewardProfile, SnakeGame

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def make_game(body, direction, food=(0, 0), seed=0):
    """Fabrique une situation artificielle pour un test ciblé."""
    game = SnakeGame(seed=seed)
    game.body = [tuple(p) for p in body]
    game.direction = direction
    game.food = tuple(food) if food is not None else None
    game.done = False
    game.won = False
    game._grow_pending = False
    return game


# ----------------------------------------------------------------------
# Départ conforme au socle
# ----------------------------------------------------------------------


def test_initial_position_matches_legacy():
    game = SnakeGame(seed=0)
    assert game.head == (rules.GRID_SIZE // 4, rules.GRID_SIZE // 2)
    assert len(game.body) == rules.INITIAL_LENGTH == 3
    assert game.direction == rules.RIGHT
    assert game.score == 0


# ----------------------------------------------------------------------
# §7 du cadrage : collision sur chacun des quatre murs
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "head, direction, action",
    [
        ((0, 7), rules.UP, LEFT),      # bord gauche
        ((14, 7), rules.UP, RIGHT),    # bord droit
        ((7, 0), rules.RIGHT, UP),     # bord haut
        ((7, 14), rules.RIGHT, DOWN),  # bord bas
    ],
)
def test_wall_collision_kills_on_every_edge(head, direction, action):
    """Sortir de la grille termine la partie, sans toucher au score officiel."""
    game = make_game([head, (head[0], head[1] + 1)], direction, food=(2, 2))
    score_before = game.score

    result = game.step(action)

    assert result.done is True
    assert result.reward == COURSE_REWARDS.death == -10.0
    assert result.score == score_before
    assert game.score == score_before
    assert result.won is False
    assert result.info["cause"] == "wall"


def test_no_more_torus_wrapping():
    """Le serpent ne réapparaît plus de l'autre côté (régression du socle)."""
    game = make_game([(0, 7), (1, 7)], rules.UP, food=(5, 5))
    game.step(LEFT)
    assert game.done is True
    assert game.head == (0, 7), "la tête ne doit pas avoir été téléportée"


# ----------------------------------------------------------------------
# Collision avec le corps
# ----------------------------------------------------------------------


def test_self_collision_kills():
    """Le serpent se mord en refermant une boucle sur lui-même."""
    body = [(5, 5), (5, 6), (6, 6), (6, 5), (7, 5)]
    game = make_game(body, rules.LEFT, food=(0, 0))
    # La tête monte puis on ferme : ici on va tout droit dans (4,5) — libre.
    # On construit plutôt une boucle serrée : tête en (6,5) direction haut.
    body = [(6, 5), (5, 5), (5, 4), (6, 4), (7, 4)]
    game = make_game(body, rules.UP, food=(0, 0))
    result = game.step(UP)  # (6,5) -> (6,4), occupé par le corps
    assert result.done is True
    assert result.reward == COURSE_REWARDS.death
    assert result.info["cause"] == "self"


def test_following_own_tail_is_legal():
    """Entrer dans la case que la queue libère au même instant est autorisé.

    C'est le comportement du socle (retrait de la queue avant le test de
    collision) et c'est la règle correcte du Snake classique.
    """
    body = [(5, 5), (5, 6), (6, 6), (6, 5)]
    game = make_game(body, rules.UP, food=(0, 0))
    result = game.step(RIGHT)  # (5,5) -> (6,5), la queue quitte (6,5)
    assert result.done is False
    assert game.head == (6, 5)


# ----------------------------------------------------------------------
# Pomme, score, croissance
# ----------------------------------------------------------------------


def test_eating_apple_adds_exactly_one_point():
    game = make_game([(5, 5), (4, 5), (3, 5)], rules.RIGHT, food=(6, 5))
    result = game.step(RIGHT)
    assert result.ate is True
    assert result.score == 1
    assert result.reward == COURSE_REWARDS.apple == 10.0
    assert result.done is False


def test_body_grows_by_one_after_apple():
    game = make_game([(5, 5), (4, 5), (3, 5)], rules.RIGHT, food=(6, 5))
    length_before = len(game.body)
    game.step(RIGHT)  # mange
    game.step(RIGHT)  # le pas suivant matérialise la croissance
    assert len(game.body) == length_before + 1


def test_length_is_constant_without_apple():
    game = make_game([(5, 5), (4, 5), (3, 5)], rules.RIGHT, food=(12, 12))
    for _ in range(4):
        game.step(RIGHT)
    assert len(game.body) == 3


def test_apple_is_never_placed_inside_the_snake():
    """Balayage sur plusieurs seeds et plusieurs longueurs de corps."""
    for seed in range(30):
        game = SnakeGame(seed=seed)
        for _ in range(200):
            assert game.food not in game.body
            legal = [i for i, ok in enumerate(game.legal_action_mask()) if ok]
            action = game._rng.choice(legal)
            if game.step(action).done:
                break


# ----------------------------------------------------------------------
# Victoire
# ----------------------------------------------------------------------


def test_victory_when_grid_is_full():
    """Dernière case libre : la manger remplit la grille et gagne la partie."""
    n = rules.GRID_SIZE
    # Serpent en serpentin couvrant toute la grille sauf une case.
    cells = []
    for y in range(n):
        row = range(n) if y % 2 == 0 else range(n - 1, -1, -1)
        cells.extend((x, y) for x in row)
    last = cells[-1]
    body_cells = cells[:-1]
    # La tête doit être adjacente à la dernière case libre.
    body = list(reversed(body_cells))
    # La tête du serpentin est adjacente à la dernière case libre ; on déduit
    # l'action du delta plutôt que de la coder en dur.
    delta = (last[0] - body[0][0], last[1] - body[0][1])
    assert abs(delta[0]) + abs(delta[1]) == 1, "tête et case libre adjacentes"

    game = make_game(body, delta, food=last)
    assert len(game.body) == rules.TOTAL_CELLS - 1

    result = game.step(rules.ACTIONS.index(delta))

    assert result.done is True
    assert result.won is True
    assert game.won is True
    assert result.reward == COURSE_REWARDS.apple + COURSE_REWARDS.victory
    assert result.score == 1
    assert game.food is None
    # Le corps occupe 224 cases et la croissance de la dernière pomme est
    # acquise : la grille est pleine, plus aucune case n'est libre.
    assert len(game.body) + 1 == rules.TOTAL_CELLS
    assert game._grow_pending is True


# ----------------------------------------------------------------------
# Masque d'actions
# ----------------------------------------------------------------------


def test_opposite_action_is_masked():
    game = make_game([(5, 5), (4, 5), (3, 5)], rules.RIGHT, food=(0, 0))
    mask = game.legal_action_mask()
    assert len(mask) == 4
    assert mask[LEFT] is False
    assert mask[RIGHT] is True and mask[UP] is True and mask[DOWN] is True


def test_opposite_action_is_ignored_not_fatal():
    """Comme dans le socle, un demi-tour demandé est ignoré, pas mortel."""
    game = make_game([(5, 5), (4, 5), (3, 5)], rules.RIGHT, food=(0, 0))
    result = game.step(LEFT)
    assert result.done is False
    assert game.direction == rules.RIGHT
    assert game.head == (6, 5)


def test_invalid_action_raises():
    game = SnakeGame(seed=0)
    with pytest.raises(ValueError):
        game.step(4)


def test_step_after_done_raises():
    game = make_game([(0, 7), (1, 7)], rules.UP, food=(5, 5))
    game.step(LEFT)
    with pytest.raises(RuntimeError):
        game.step(UP)


# ----------------------------------------------------------------------
# Récompenses (§8 du cadrage)
# ----------------------------------------------------------------------


def test_reward_profile_matches_the_slides():
    assert COURSE_REWARDS.apple == 10.0
    assert COURSE_REWARDS.death == -10.0
    assert COURSE_REWARDS.step == 0.1
    assert COURSE_REWARDS.victory == 100.0


def test_plain_move_rewards_step_value():
    game = make_game([(5, 5), (4, 5), (3, 5)], rules.RIGHT, food=(12, 12))
    result = game.step(RIGHT)
    assert result.reward == COURSE_REWARDS.step
    assert result.score == 0


def test_reward_profile_is_swappable_without_touching_the_score():
    """Un profil expérimental change le signal d'apprentissage, pas le score."""
    profile = RewardProfile(name="experimental", apple=1.0, death=-1.0, step=0.0)
    game = SnakeGame(seed=0, reward_profile=profile)
    game.body = [(5, 5), (4, 5), (3, 5)]
    game.food = (6, 5)
    result = game.step(RIGHT)
    assert result.reward == 1.0
    assert result.score == 1, "le score officiel reste le nombre de pommes"


# ----------------------------------------------------------------------
# Déterminisme
# ----------------------------------------------------------------------


def test_same_seed_gives_identical_games():
    def play(seed):
        game = SnakeGame(seed=seed)
        trace = [game.food]
        for action in [RIGHT, DOWN, RIGHT, DOWN, RIGHT, UP, RIGHT]:
            if game.done:
                break
            game.step(action)
            trace.append(game.food)
        return trace

    assert play(123) == play(123)


def test_different_seeds_give_different_food_sequences():
    assert SnakeGame(seed=1).food != SnakeGame(seed=2).food


# ----------------------------------------------------------------------
# Troncature expérimentale
# ----------------------------------------------------------------------


def test_truncation_is_off_by_default():
    game = SnakeGame(seed=0)
    assert game.max_steps_without_food is None


def test_truncation_marks_truncated_not_lost():
    game = SnakeGame(seed=0, max_steps_without_food=3)
    game.body = [(5, 5), (4, 5), (3, 5)]
    game.food = (12, 12)
    game.step(RIGHT)
    game.step(RIGHT)
    result = game.step(RIGHT)
    assert result.done is True
    assert result.truncated is True
    assert result.won is False
    assert result.reward == COURSE_REWARDS.step, "pas de pénalité de mort"


# ----------------------------------------------------------------------
# Snapshot pour les replays
# ----------------------------------------------------------------------


def test_snapshot_is_json_serialisable_and_complete():
    import json

    game = SnakeGame(seed=7)
    snap = game.snapshot(action=RIGHT, reward=0.1)
    for key in ("head", "body", "food", "direction", "action", "reward", "score",
                "steps", "done"):
        assert key in snap
    json.dumps(snap)  # ne doit pas lever
