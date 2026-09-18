"""
Q-learning with tile coding for Bomberman (training side).

Learning only happens when the framework runs the agent with --train. The
weights approximated by the tile coder live on the shared 'self' object that
callbacks.py also uses, so callbacks.py never needs to know about training.

Core pieces:
  * Off-policy Q-learning over the tile-coded linear value approximation.
    The TD target uses max_a' Q(s', a') restricted to valid actions.
  * Uniform experience replay to decorrelate successive updates and to
    stabilise the bootstrapping against the tile-coded approximator.
  * Potential-based reward shaping (Ng et al., ICML 1999) so that the agent
    receives a dense signal while the optimal policy is preserved, plus
    auxiliary rewards for the in-game events provided by the environment.
"""
import pickle
import random
from collections import deque

import numpy as np

import events as e
import settings as s

from .callbacks import (
    ACTIONS, ACTION_INDICES, MODEL_FILE, NUM_ACTIONS, NUM_TILINGS, MAX_DIST,
    prepare_state, features_for_action, tile_indices, valid_actions,
)

TRAINING_RESULTS_FILE = 'training_results.pkl'

INITIAL_ALPHA = 0.03
ALPHA_MIN = 0.002
ALPHA_DECAY = 0.9998
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.998
GAMMA = 0.95
STEP_PENALTY = -0.01
REPLAY_BUFFER_SIZE = 20000
REPLAY_BATCH_SIZE = 32
LOG_EVERY = 25

POTENTIAL_COIN = 2.0
POTENTIAL_CRATE = 0.0
POTENTIAL_DANGER = 0.3
POTENTIAL_EXPLOSION = 0.5
POTENTIAL_OPPONENT = 0.5

RETRO_WINDOW = s.BOMB_TIMER + 1
CRATE_RETRO = 0.0
KILL_RETRO = 4.0

EVENT_REWARDS = {
    e.COIN_COLLECTED: 2.0,
    e.KILLED_OPPONENT: 8.0,
    e.CRATE_DESTROYED: 0.0,
    e.COIN_FOUND: 0.2,
    e.INVALID_ACTION: -0.3,
    e.KILLED_SELF: -6.0,
    e.GOT_KILLED: -5.0,
    e.OPPONENT_ELIMINATED: 0.5,
    e.SURVIVED_ROUND: 2.0,
}


def _transition(tiles_sa, action_idx, reward, tiles_next, next_valid, terminal):
    return {
        'tiles': list(tiles_sa),
        'action_idx': int(action_idx),
        'reward': float(reward),
        'tiles_next': None if tiles_next is None else [list(t) for t in tiles_next],
        'next_valid': None if next_valid is None else tuple(bool(v) for v in next_valid),
        'terminal': bool(terminal),
    }


def reward_from_events(events):
    total = 0.0
    for event in events:
        total += EVENT_REWARDS.get(event, 0.0)
    return total


def _norm_dist(distance):
    return 1.0 if np.isinf(distance) else float(distance) / MAX_DIST


def potential(view):
    if view is None:
        return 0.0
    x, y = view.pos
    value = 0.0
    value -= POTENTIAL_COIN * _norm_dist(float(view.coin_dist[x, y]))
    value -= POTENTIAL_CRATE * _norm_dist(float(view.bomb_dist[x, y]))
    value -= POTENTIAL_DANGER * min(float(view.danger[x, y]), 1.0)
    if float(view.explosion_map[x, y]) > 0.0:
        value -= POTENTIAL_EXPLOSION
    if view.opp_set:
        value += POTENTIAL_OPPONENT * (1.0 - _norm_dist(float(view.opp_dist[x, y])))
    return value


def _clear_inputs(self):
    self.episode_reward = 0.0
    self.episode_count = 0
    self.ep_coins = 0
    self.ep_kills = 0
    self.ep_crates = 0
    self.ep_death_cause = None
    self.ep_event_reward = 0.0
    self.last_bomb_transition = None
    self.last_bomb_step = -RETRO_WINDOW


def setup_training(self):
    self.gamma = GAMMA
    if not hasattr(self, 'alpha'):
        self.alpha = INITIAL_ALPHA
    if not hasattr(self, 'epsilon'):
        self.epsilon = EPSILON_START
    self.replay = deque(maxlen=REPLAY_BUFFER_SIZE)
    _clear_inputs(self)
    self.training_history = {
        'rewards': [],
        'lengths': [],
        'deaths': [],
        'death_causes': [],
        'coins': [],
        'kills': [],
        'crates': [],
        'event_rewards': [],
    }


def _note_events(self, events):
    for event in events:
        if event == e.COIN_COLLECTED:
            self.ep_coins += 1
        elif event == e.KILLED_OPPONENT:
            self.ep_kills += 1
        elif event == e.CRATE_DESTROYED:
            self.ep_crates += 1
        elif event == e.KILLED_SELF:
            self.ep_death_cause = 'KILLED_SELF'
        elif event == e.GOT_KILLED and self.ep_death_cause != 'KILLED_SELF':
            self.ep_death_cause = 'GOT_KILLED'


def _learn_from_replay(self):
    if len(self.replay) < REPLAY_BATCH_SIZE:
        return
    batch = self.rng.sample(self.replay, REPLAY_BATCH_SIZE)
    per_tile_alpha = self.alpha / NUM_TILINGS
    for trans in batch:
        tiles_sa = trans['tiles']
        action_idx = trans['action_idx']
        reward = trans['reward']
        tiles_next = trans['tiles_next']
        next_valid = trans['next_valid']
        terminal = trans['terminal']
        q_value = float(self.model_weights[tiles_sa].sum())
        if terminal or tiles_next is None:
            target = reward
        else:
            best_next = -np.inf
            for i in range(NUM_ACTIONS):
                if not next_valid[i]:
                    continue
                value = float(self.model_weights[tiles_next[i]].sum())
                if value > best_next:
                    best_next = value
            target = reward + self.gamma * best_next if best_next > -np.inf else reward
        delta = target - q_value
        np.add.at(self.model_weights, tiles_sa, per_tile_alpha * delta)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    old_view = prepare_state(old_game_state)
    if old_view is None:
        return
    new_view = prepare_state(new_game_state)

    reward = reward_from_events(events) + STEP_PENALTY
    reward += self.gamma * potential(new_view) - potential(old_view)
    self.ep_event_reward += reward_from_events(events)

    tiles_sa = tile_indices(self.tile_coder, old_view, self_action)
    action_idx = ACTION_INDICES[self_action]

    if new_view is None:
        return

    next_valid = valid_actions(new_view)
    tiles_next = [tile_indices(self.tile_coder, new_view, action) for action in ACTIONS]
    trans = _transition(tiles_sa, action_idx, reward, tiles_next, next_valid, False)

    current_step = int(new_game_state['step'])
    if self.last_bomb_transition is not None \
            and current_step - self.last_bomb_step <= RETRO_WINDOW:
        if e.CRATE_DESTROYED in events:
            self.last_bomb_transition['reward'] += CRATE_RETRO
        if e.KILLED_OPPONENT in events:
            self.last_bomb_transition['reward'] += KILL_RETRO

    self.replay.append(trans)
    if self_action == 'BOMB':
        self.last_bomb_transition = trans
        self.last_bomb_step = current_step

    self.episode_reward += reward
    _note_events(self, events)
    _learn_from_replay(self)


def end_of_round(self, last_game_state, last_action, events):
    reward = reward_from_events(events)
    died = e.KILLED_SELF in events or e.GOT_KILLED in events

    if died and last_game_state is not None:
        last_view = prepare_state(last_game_state)
        if last_view is not None:
            tiles_sa = tile_indices(self.tile_coder, last_view, last_action)
            action_idx = ACTION_INDICES[last_action]
            self.replay.append(_transition(tiles_sa, action_idx, reward, None, None, True))

    _note_events(self, events)
    self.episode_reward += reward
    self.episode_count += 1

    self.training_history['rewards'].append(self.episode_reward)
    self.training_history['lengths'].append(
        int(last_game_state['step']) if last_game_state is not None else 0)
    self.training_history['deaths'].append(int(died))
    self.training_history['death_causes'].append(self.ep_death_cause)
    self.training_history['coins'].append(self.ep_coins)
    self.training_history['kills'].append(self.ep_kills)
    self.training_history['crates'].append(self.ep_crates)
    self.training_history['event_rewards'].append(self.ep_event_reward)

    _learn_from_replay(self)

    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.alpha = max(ALPHA_MIN, self.alpha * ALPHA_DECAY)
    _clear_inputs(self)

    if self.episode_count % LOG_EVERY == 0:
        self.logger.info(
            'Episode %d: reward=%.2f events=%.2f length=%d died=%d cause=%s epsilon=%.2f alpha=%.4f '
            'coins=%d kills=%d crates=%d',
            self.episode_count,
            self.training_history['rewards'][-1],
            self.training_history['event_rewards'][-1],
            self.training_history['lengths'][-1],
            self.training_history['deaths'][-1],
            self.training_history['death_causes'][-1],
            self.epsilon,
            self.alpha,
            self.training_history['coins'][-1],
            self.training_history['kills'][-1],
            self.training_history['crates'][-1],
        )

    with open(MODEL_FILE, 'wb') as file:
        pickle.dump({
            'weights': self.model_weights,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
        }, file)

    with open(TRAINING_RESULTS_FILE, 'wb') as file:
        pickle.dump(self.training_history, file)