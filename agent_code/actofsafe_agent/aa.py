import pickle
import numpy as np
import matplotlib.pyplot as plt


# Load results
with open("c:/Users/msı/Desktop/Universitat Heidelberg/MLE/bomberman/bomberman_rl/agent_code/actofsafe_agent/training_results.pkl", "rb") as file:
    results = pickle.load(file)


rewards = np.array(results["rewards"])
episode_lengths = np.array(results["episode_lengths"])
deaths = np.array(results["deaths"])

window = 100

reward_moving_average = np.convolve(
    rewards,
    np.ones(window) / window,
    mode="valid"
)

length_moving_average = np.convolve(
    episode_lengths,
    np.ones(window) / window,
    mode="valid"
)
plt.figure(figsize=(10, 5))

plt.plot(
    reward_moving_average
)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("100-Episode Moving Average Reward")

plt.grid(True)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 5))

plt.plot(
    length_moving_average
)

plt.xlabel("Episode")
plt.ylabel("Average Survival Length")
plt.title("100-Episode Moving Average Survival Length")

plt.grid(True)
plt.tight_layout()
plt.show()