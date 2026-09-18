"""
Tile-coded Q-learning agent for Bomberman.

This module implements the policy-evaluation / action-selection core of the
agent: a hashed tile coder (Sutton & Barto, 'Reinforcement Learning: An
Introduction', Section 9.5.4) over a hand-crafted feature vector, plus the
feature engineering that turns a raw game_state into a compact, bounded
description of the situation around the agent.

The weight vector is shared with train.py: callbacks.py is responsible for
representing states and choosing actions (playing), train.py is responsible
for learning (Q-learning updates) and for persisting the learned weights.
"""
import pickle
import random
from collections import deque

import numpy as np

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']
ACTION_INDICES = {action: i for i, action in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)

MODEL_FILE = 'my-saved-model.pt'
FINE_TUNE_EPSILON = 0.2
FINE_TUNE_ALPHA = 0.01
COIN_SAFE_RADIUS = 3
CRATE_INTENT_BONUS = 1.5
KILL_INTENT_BONUS = 1.5
COIN_INTENT_BONUS = 2.0

# Tile coder configuration (keep identical between training and playing,
# otherwise the learned weights would point to different features).
NUM_TILINGS = 16
TABLE_SIZE = 1 << 16
FEATURE_BINS = (2, 2, 2, 2, 8, 2, 8, 8, 8, 8, 2, 2, 2, 4, 2, 8)
HASH_MASK = (1 << 64) - 1

MAX_DIST = (s.COLS - 2) + (s.ROWS - 2)
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
PROSPECTIVE_DANGER = (s.BOMB_TIMER - (s.BOMB_TIMER - 1)) / s.BOMB_TIMER


def _clamp01(value):
    if value < 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0 - 1e-9
    return float(value)


def _hash_coords(coords):
    h = 1469598103934665603
    for c in coords:
        h = ((h ^ int(c)) * 1099511628211) & HASH_MASK
    h ^= h >> 33
    h = (h * 0xFF51AFD7ED558CCD) & HASH_MASK
    h ^= h >> 33
    h = (h * 0xC4CEB9FE1A85EC53) & HASH_MASK
    h ^= h >> 33
    return h


class TileCoder:
    """
    Hash-based tile coding over a bounded feature vector.

    Each feature dimension d is partitioned into FEATURE_BINS[d] coarse tiles.
    NUM_TILINGS mutually offset tilings are overlaid on top of each other, so
    that every state activates exactly one tile from every tiling. The active
    tile coordinates are combined with the action index and hashed down into a
    table of fixed size, yielding a sparse binary feature representation that
    can be paired with a weight vector to form a linear value approximator.
    """

    def __init__(self, bins_per_dim, num_tilings, table_size):
        self.bins = np.asarray(bins_per_dim, dtype=np.int64)
        self.num_tilings = int(num_tilings)
        self.table_size = int(table_size)
        self._mask = self.table_size - 1

    def encode(self, features, action_idx):
        indices = []
        num_tilings = self.num_tilings
        for tiling in range(num_tilings):
            tiling_x2 = tiling * 2
            b = tiling
            coords = [tiling]
            for d in range(self.bins.size):
                scaled = float(features[d]) * float(self.bins[d]) * float(num_tilings)
                q = int(scaled)
                coords.append((q + b) // num_tilings)
                b += tiling_x2
            coords.append(action_idx)
            indices.append(_hash_coords(coords) & self._mask)
        return indices


class WorldView:
    """Precomputed, action-independent information about the current state."""

    __slots__ = (
        'field', 'explosion_map', 'danger', 'walkable',
        'bomb_set', 'bombs', 'opp_set', 'pos', 'bomb_avail', 'step',
        'coin_dist', 'bomb_dist', 'opp_dist', 'lethal', 'lethal_hard',
    )

    def __init__(self):
        for key in self.__slots__:
            setattr(self, key, None)

    def copy(self):
        other = WorldView()
        for key in self.__slots__:
            setattr(other, key, getattr(self, key))
        return other


def multi_source_bfs(walkable, starts):
    x_lim, y_lim = walkable.shape
    dist = np.full(walkable.shape, np.inf)
    queue = deque()
    for x, y in starts:
        x, y = int(x), int(y)
        if 0 <= x < x_lim and 0 <= y < y_lim and walkable[x, y] and np.isinf(dist[x, y]):
            dist[x, y] = 0.0
            queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        next_dist = dist[x, y] + 1.0
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < x_lim and 0 <= ny < y_lim and walkable[nx, ny] and np.isinf(dist[nx, ny]):
                dist[nx, ny] = next_dist
                queue.append((nx, ny))
    return dist


def danger_map(field, bombs):
    x_lim, y_lim = field.shape
    danger = np.zeros(field.shape)
    for (bx, by), t in bombs:
        value = (s.BOMB_TIMER - int(t)) / s.BOMB_TIMER
        if value > danger[bx, by]:
            danger[bx, by] = value
        for dx, dy in DIRECTIONS:
            for i in range(1, s.BOMB_POWER + 1):
                nx, ny = bx + dx * i, by + dy * i
                if not (0 <= nx < x_lim and 0 <= ny < y_lim):
                    break
                if field[nx, ny] == -1:
                    break
                if value > danger[nx, ny]:
                    danger[nx, ny] = value
    return danger


def blast_cells(field, x, y):
    x_lim, y_lim = field.shape
    cells = [(x, y)]
    for dx, dy in DIRECTIONS:
        for i in range(1, s.BOMB_POWER + 1):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < x_lim and 0 <= ny < y_lim):
                break
            if field[nx, ny] == -1:
                break
            cells.append((nx, ny))
    return cells


def bombable_cells(field):
    x_lim, y_lim = field.shape
    cells = set()
    for cx, cy in np.argwhere(field == 1):
        cx, cy = int(cx), int(cy)
        for dx, dy in DIRECTIONS:
            for i in range(1, s.BOMB_POWER + 1):
                nx, ny = cx + dx * i, cy + dy * i
                if not (0 <= nx < x_lim and 0 <= ny < y_lim):
                    break
                if field[nx, ny] == -1:
                    break
                cells.add((nx, ny))
    return cells


def prepare_state(game_state):
    if game_state is None:
        return None

    field = game_state['field']
    _, _, bomb_avail, pos = game_state['self']
    bombs = game_state['bombs']
    others = game_state['others']
    walkable = field == 0

    view = WorldView()
    view.field = field
    view.danger = danger_map(field, bombs)
    view.explosion_map = game_state['explosion_map']
    view.walkable = walkable
    view.bomb_set = {xy for xy, _ in bombs}
    view.bombs = list(bombs)
    view.opp_set = {other[3] for other in others}
    view.pos = pos
    view.bomb_avail = bool(bomb_avail)
    view.step = int(game_state['step'])
    view.coin_dist = multi_source_bfs(walkable, list(game_state['coins']))
    bomb_cells = {c for c in bombable_cells(field) if field[c] == 0}
    view.bomb_dist = multi_source_bfs(walkable, sorted(bomb_cells))
    view.opp_dist = multi_source_bfs(walkable, list(view.opp_set))

    lethal = set()
    lethal_hard = set()
    for (bx, by), tick in bombs:
        lethal.update(blast_cells(field, bx, by))
        if (s.BOMB_TIMER - int(tick)) / s.BOMB_TIMER >= 0.5:
            lethal_hard.update(blast_cells(field, bx, by))
    for (ex, ey), value in np.ndenumerate(view.explosion_map):
        if value > 0.0:
            lethal.add((int(ex), int(ey)))
            lethal_hard.add((int(ex), int(ey)))
    view.lethal = lethal
    view.lethal_hard = lethal_hard
    return view


def _open_neighbor(view, x, y, dx, dy):
    x_lim, y_lim = view.field.shape
    nx, ny = x + dx, y + dy
    if not (0 <= nx < x_lim and 0 <= ny < y_lim):
        return 0.0
    if view.field[nx, ny] != 0:
        return 0.0
    if (nx, ny) in view.bomb_set or (nx, ny) in view.opp_set:
        return 0.0
    return 1.0


def features_for_action(view, action):
    x, y = view.pos
    dest_x, dest_y = x, y
    if action == 'UP':
        dest_y = y - 1
    elif action == 'DOWN':
        dest_y = y + 1
    elif action == 'LEFT':
        dest_x = x - 1
    elif action == 'RIGHT':
        dest_x = x + 1

    x_lim, y_lim = view.field.shape
    cx = min(max(dest_x, 0), x_lim - 1)
    cy = min(max(dest_y, 0), y_lim - 1)

    prospective_blast = None
    if action == 'BOMB':
        prospective_blast = set(blast_cells(view.field, cx, cy))

    open_up = _open_neighbor(view, cx, cy, 0, -1)
    open_down = _open_neighbor(view, cx, cy, 0, 1)
    open_left = _open_neighbor(view, cx, cy, -1, 0)
    open_right = _open_neighbor(view, cx, cy, 1, 0)

    danger = _clamp01(float(view.danger[cx, cy]))
    if prospective_blast is not None and (cx, cy) in prospective_blast:
        danger = max(danger, PROSPECTIVE_DANGER)

    in_explosion = 1.0 if float(view.explosion_map[cx, cy]) > 0.0 else 0.0

    escape = 0.0
    for dx, dy in DIRECTIONS:
        if _open_neighbor(view, cx, cy, dx, dy) == 0.0:
            continue
        nx, ny = cx + dx, cy + dy
        if prospective_blast is not None and (nx, ny) in prospective_blast:
            continue
        if float(view.danger[nx, ny]) > 0.0:
            continue
        if float(view.explosion_map[nx, ny]) > 0.0:
            continue
        escape += 1.0
    escape = escape / 4.0

    coin_d = float(view.coin_dist[cx, cy])
    coin_dist = 1.0 if np.isinf(coin_d) else coin_d / MAX_DIST
    coin_dist = _clamp01(coin_dist)

    bomb_d = float(view.bomb_dist[cx, cy])
    bomb_dist = 1.0 if np.isinf(bomb_d) else bomb_d / MAX_DIST
    bomb_dist = _clamp01(bomb_dist)

    if view.opp_set:
        opp_d = float(view.opp_dist[cx, cy])
        opp_dist = 1.0 if np.isinf(opp_d) else opp_d / MAX_DIST
        opp_dist = _clamp01(opp_dist)
    else:
        opp_dist = 1.0

    bomb_avail = 1.0 if view.bomb_avail else 0.0

    adj_crate = 0.0
    adj_opp = 0.0
    for dx, dy in DIRECTIONS:
        nx, ny = cx + dx, cy + dy
        if 0 <= nx < x_lim and 0 <= ny < y_lim:
            if view.field[nx, ny] == 1:
                adj_crate = 1.0
            if (nx, ny) in view.opp_set:
                adj_opp = 1.0

    bomb_crates = 0.0
    bomb_kills = 0.0
    if action == 'BOMB':
        for bx, by in prospective_blast:
            if view.field[bx, by] == 1:
                bomb_crates += 1.0
            if (bx, by) in view.opp_set:
                bomb_kills = 1.0
        bomb_crates = _clamp01(bomb_crates / s.BOMB_POWER)

    step_frac = _clamp01(view.step / s.MAX_STEPS)

    features = np.asarray([
        open_up, open_down, open_left, open_right,
        danger, in_explosion, escape,
        coin_dist, bomb_dist, opp_dist,
        bomb_avail, adj_crate, adj_opp,
        bomb_crates, bomb_kills, step_frac,
    ], dtype=np.float64)
    features[features < 0.0] = 0.0
    features[features >= 1.0] = 1.0 - 1e-9
    return features


def state_to_features(game_state, action):
    view = prepare_state(game_state)
    if view is None:
        return None
    return features_for_action(view, action)


def action_delta(action):
    if action == 'UP':
        return 0, -1
    if action == 'DOWN':
        return 0, 1
    if action == 'LEFT':
        return -1, 0
    if action == 'RIGHT':
        return 1, 0
    return 0, 0


def can_escape_own_bomb(view, x, y):
    x_lim, y_lim = view.field.shape
    cross = set(blast_cells(view.field, x, y))
    frontier = {(x, y)}
    seen = {(x, y)}
    for _ in range(1, s.BOMB_TIMER):
        next_frontier = set()
        for cx, cy in frontier:
            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < x_lim and 0 <= ny < y_lim):
                    continue
                if view.field[nx, ny] != 0:
                    continue
                if (nx, ny) in view.bomb_set or (nx, ny) in view.opp_set:
                    continue
                if (nx, ny) in view.lethal_hard:
                    continue
                if (nx, ny) not in cross:
                    return True
                if (nx, ny) not in seen:
                    seen.add((nx, ny))
                    next_frontier.add((nx, ny))
        frontier = next_frontier
        if not frontier:
            return False
    return False


def blast_time_map(view):
    times = np.full(view.field.shape, np.inf)
    times[view.explosion_map > 0.0] = 0.0
    for (bx, by), tick in view.bombs:
        blast_at = float(max(int(tick), 0))
        for cx, cy in blast_cells(view.field, bx, by):
            if blast_at < times[cx, cy]:
                times[cx, cy] = blast_at
    return times


def in_blast(view, x, y):
    return (x, y) in view.lethal


def escape_step(view, x, y):
    times = blast_time_map(view)
    if not in_blast(view, x, y):
        return None

    x_lim, y_lim = view.field.shape
    horizon = int(s.BOMB_TIMER + s.EXPLOSION_TIMER + 2)
    visited = {(x, y, 0)}
    queue = deque()

    def passable(nx, ny, t):
        if not (0 <= nx < x_lim and 0 <= ny < y_lim):
            return None
        if view.field[nx, ny] != 0:
            return None
        if (nx, ny) in view.bomb_set or (nx, ny) in view.opp_set:
            return None
        if t >= times[nx, ny]:
            return None
        return times[nx, ny]

    for action in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
        dx, dy = action_delta(action)
        nx, ny = x + dx, y + dy
        remaining = passable(nx, ny, 1)
        if remaining is None:
            continue
        if np.isinf(remaining):
            return action
        state = (nx, ny, 1)
        if state not in visited:
            visited.add(state)
            queue.append((nx, ny, 1, action))

    while queue:
        cx, cy, t, first = queue.popleft()
        if t >= horizon:
            continue
        for action in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
            dx, dy = action_delta(action)
            nx, ny = cx + dx, cy + dy
            remaining = passable(nx, ny, t + 1)
            if remaining is None:
                continue
            if np.isinf(remaining):
                return first
            state = (nx, ny, t + 1)
            if state not in visited:
                visited.add(state)
                queue.append((nx, ny, t + 1, first))
        if t + 1 < times[cx, cy]:
            state = (cx, cy, t + 1)
            if state not in visited:
                visited.add(state)
                queue.append((cx, cy, t + 1, first))
    return None


def valid_actions(view):
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    x_lim, y_lim = view.field.shape
    x, y = view.pos
    hard = view.lethal_hard

    def blocked(xx, yy):
        if not (0 <= xx < x_lim and 0 <= yy < y_lim):
            return True
        if view.field[xx, yy] != 0:
            return True
        if (xx, yy) in view.bomb_set or (xx, yy) in view.opp_set:
            return True
        return False

    for i, action in enumerate(ACTIONS):
        if action == 'WAIT':
            mask[i] = (x, y) not in hard
        elif action == 'BOMB':
            if not view.bomb_avail:
                continue
            if (x, y) in hard:
                continue
            if in_blast(view, x, y):
                continue
            mask[i] = can_escape_own_bomb(view, x, y)
        else:
            dx, dy = action_delta(action)
            nx, ny = x + dx, y + dy
            mask[i] = not blocked(nx, ny) and (nx, ny) not in hard
    return mask


def tile_indices(tile_coder, view, action):
    return tile_coder.encode(features_for_action(view, action), ACTION_INDICES[action])


def _load_model():
    try:
        with open(MODEL_FILE, 'rb') as file:
            return pickle.load(file)
    except (OSError, EOFError, TypeError, AttributeError, KeyError, pickle.PickleError):
        return None


def setup(self):
    self.rng = random.Random()
    self.tile_coder = TileCoder(FEATURE_BINS, NUM_TILINGS, TABLE_SIZE)

    model = _load_model()
    if self.train:
        if model is not None and isinstance(model.get('weights'), np.ndarray) \
                and model['weights'].shape == (TABLE_SIZE,):
            self.model_weights = model['weights'].astype(np.float64)
            self.epsilon = max(float(model.get('epsilon', 1.0)), FINE_TUNE_EPSILON)
            self.alpha = max(float(model.get('alpha', 0.03)), FINE_TUNE_ALPHA)
            self.logger.info('Continuing training from previous weights.')
        else:
            self.model_weights = np.zeros(TABLE_SIZE, dtype=np.float64)
            self.epsilon = 1.0
            self.alpha = 0.03
            self.logger.info('Initialising fresh tile-coded weight table.')
    else:
        if model is not None and isinstance(model.get('weights'), np.ndarray) \
                and model['weights'].shape == (TABLE_SIZE,):
            self.model_weights = model['weights'].astype(np.float64)
            self.logger.info('Loaded trained weight table.')
        else:
            self.model_weights = np.zeros(TABLE_SIZE, dtype=np.float64)
            self.logger.warning('No trained model found - using untrained weights.')
        self.epsilon = 0.0


def danger_within(view, x, y, radius):
    for bx, by in view.bomb_set:
        if abs(bx - x) + abs(by - y) <= radius:
            return True
    x_lim, y_lim = view.field.shape
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) > radius:
                continue
            nx, ny = x + dx, y + dy
            if not (0 <= nx < x_lim and 0 <= ny < y_lim):
                continue
            if view.danger[nx, ny] > 0.0 or view.explosion_map[nx, ny] > 0.0:
                return True
    return False


def nearest_danger_dist(view, gx, gy):
    best = float(MAX_DIST + s.BOMB_POWER)
    cells = np.nonzero((view.danger > 0.0) | (view.explosion_map > 0.0))
    for cx, cy in zip(cells[0].tolist(), cells[1].tolist()):
        dist = abs(cx - gx) + abs(cy - gy)
        if dist < best:
            best = dist
    for bx, by in view.bomb_set:
        dist = abs(bx - gx) + abs(by - gy)
        if dist < best:
            best = dist
    return best


def open_neighbors(view, x, y):
    x_lim, y_lim = view.field.shape
    count = 0
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < x_lim and 0 <= ny < y_lim and view.field[nx, ny] == 0:
            count += 1
    return count


def flee_choice(view, x, y, valid):
    best_action = None
    best_key = None
    for action in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
        idx = ACTION_INDICES[action]
        if not valid[idx]:
            continue
        dx, dy = action_delta(action)
        nx, ny = x + dx, y + dy
        key = (nearest_danger_dist(view, nx, ny), open_neighbors(view, nx, ny))
        if best_key is None or key > best_key:
            best_key = key
            best_action = action
    return best_action


def act(self, game_state):
    if game_state is None:
        return 'WAIT'

    view = prepare_state(game_state)
    x_lim, y_lim = view.field.shape
    x, y = view.pos
    valid = valid_actions(view)

    if in_blast(view, x, y):
        escape = escape_step(view, x, y)
        if escape is not None:
            return escape
        fallback = flee_choice(view, x, y, valid)
        if fallback is not None:
            return fallback

    intent_bonus = np.zeros(NUM_ACTIONS)

    if view.bomb_avail:
        bomb_target = False
        kill_target = False
        for bx, by in blast_cells(view.field, x, y):
            if view.field[bx, by] == 1:
                bomb_target = True
            if (bx, by) in view.opp_set:
                kill_target = True
        if kill_target:
            intent_bonus[ACTION_INDICES['BOMB']] += KILL_INTENT_BONUS
        elif bomb_target:
            intent_bonus[ACTION_INDICES['BOMB']] += CRATE_INTENT_BONUS

    if not danger_within(view, x, y, COIN_SAFE_RADIUS):
        coin_here = float(view.coin_dist[x, y])
        if np.isfinite(coin_here):
            for action in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
                dx, dy = action_delta(action)
                nx, ny = x + dx, y + dy
                if not (0 <= nx < x_lim and 0 <= ny < y_lim):
                    continue
                if view.field[nx, ny] != 0:
                    continue
                next_dist = float(view.coin_dist[nx, ny])
                if np.isfinite(next_dist) and next_dist < coin_here:
                    intent_bonus[ACTION_INDICES[action]] += COIN_INTENT_BONUS

    if self.train and self.rng.random() < self.epsilon:
        candidates = np.flatnonzero(valid)
        if candidates.size == 0:
            return 'WAIT'
        choice = int(self.rng.choice(candidates))
        return ACTIONS[choice]

    q_values = np.zeros(NUM_ACTIONS)
    for i, action in enumerate(ACTIONS):
        tiles = self.tile_coder.encode(features_for_action(view, action), i)
        q_values[i] = float(self.model_weights[tiles].sum())

    scored = q_values + intent_bonus
    masked = np.where(valid, scored, -np.inf)
    choice = int(np.argmax(masked))
    return ACTIONS[choice]