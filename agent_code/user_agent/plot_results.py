"""
Plot the training history recorded by train.py.

Produces two separate figures, each in its own window:
  * Rewards: raw per-episode reward with a rolling average overlay.
  * Survival: rolling survival rate and episode length (steps survived).

The script reads 'training_results.pkl' from the agent directory (where
train.py saves it), so it works no matter where it is invoked from.
"""
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = 'training_results.pkl'
WINDOW = 50


def load_results(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def smooth(values, window):
    values = np.asarray(values, dtype=float)
    if window < 2 or values.size < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode='valid')


def main():
    path = Path(__file__).resolve().parent / RESULTS_FILE
    if not path.is_file():
        print(f'No training history found at {path}. Run the agent in --train mode first.')
        sys.exit(1)

    history = load_results(path)
    rewards = np.asarray(history.get('rewards', []), dtype=float)
    lengths = np.asarray(history.get('lengths', []), dtype=float)
    deaths = np.asarray(history.get('deaths', []), dtype=float)
    coins = np.asarray(history.get('coins', []), dtype=float)
    crates = np.asarray(history.get('crates', []), dtype=float)
    kills = np.asarray(history.get('kills', []), dtype=float)
    if rewards.size == 0:
        print('Training history is empty.')
        sys.exit(1)

    episodes = np.arange(1, rewards.size + 1)
    window = min(WINDOW, rewards.size)

    rewards_fig = plt.figure('Training rewards')
    plt.plot(episodes, rewards, color='tab:blue', alpha=0.35, label='reward / episode')
    if window >= 2:
        smoothed = smooth(rewards, window)
        offset = rewards.size - smoothed.size
        plt.plot(episodes[offset:], smoothed, color='tab:blue', lw=2,
                 label=f'rolling average ({window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Bomberman agent: reward per episode')
    plt.legend()
    plt.grid(alpha=0.3)

    survival_fig = plt.figure('Survival')
    score_rate = coins + 0.5 * crates + 2.0 * kills
    survived = 1.0 - deaths

    plt.subplot(2, 1, 1)
    plt.plot(episodes, score_rate, color='tab:green', alpha=0.35, label='scoring rate / episode')
    if window >= 2:
        smoothed_score = smooth(score_rate, window)
        offset = score_rate.size - smoothed_score.size
        plt.plot(episodes[offset:], smoothed_score, color='tab:green', lw=2,
                 label=f'rolling average ({window} episodes)')
    plt.ylabel('Score rate')
    plt.title('Bomberman agent: scoring (coins + 0.5*crates + 2*kills)')
    plt.legend()
    plt.grid(alpha=0.3)

    ax_steps = plt.subplot(2, 1, 2)
    ax_steps.plot(episodes, lengths, color='tab:purple', alpha=0.5, label='steps survived / episode')
    ax_steps.set_xlabel('Episode')
    ax_steps.set_ylabel('Steps survived')
    ax_steps.grid(alpha=0.3)
    ax_surv = ax_steps.twinx()
    if window >= 2:
        smoothed_survival = smooth(survived, window)
        offset = survived.size - smoothed_survival.size
        ax_surv.plot(episodes[offset:], smoothed_survival * 100.0, color='tab:red', lw=2,
                     label=f'survival rate % ({window} episodes)')
    ax_surv.set_ylabel('Survival rate (%)')
    ax_surv.set_ylim(0, 100)
    handles = ax_steps.get_lines() + ax_surv.get_lines()
    ax_steps.legend(handles, [item.get_label() for item in handles])
    ax_steps.set_title('Bomberman agent: survival')

    plt.show()


if __name__ == '__main__':
    main()