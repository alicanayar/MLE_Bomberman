import pickle
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
BOMB_NEXT_TO_CRATE = "BOMB_NEXT_TO_CRATE"
BOMB_NOT_NEXT_TO_CRATE = "BOMB_NOT_NEXT_TO_CRATE"


def setup_training(self):

    self.alpha = 0.01
    self.gamma = 0.95

    self.epsilon = 1.0

    self.episode_reward = 0
    self.episode_length = 0

    self.training_rewards = []
    self.training_lengths = []
    self.training_deaths = []




def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    #Calculate Reward
    reward = reward_from_events(self,events)

    #store data
    self.episode_reward += reward
    self.episode_length += 1


    #Calculate current state's Q(s,a)
    phi = state_to_features(old_game_state,self_action)
    current_q = np.dot(self.weights, phi)

    #Calculate next state's Q(s,a)
    next_q = max(np.dot(self.weights, state_to_features(new_game_state, action)) for action in ACTIONS)

    #Calculate target
    target = reward + self.gamma * next_q

    #Error
    error = target - current_q

    #custom event for reward dropping a bomb next to crate and penalty dropping a bomb any placa than next to crate 
    if self_action == "BOMB": 
        if phi[-3] == 1:  #phi[-3] is crate_bombing feature
            events.append(BOMB_NEXT_TO_CRATE)
        else:
            events.append(BOMB_NOT_NEXT_TO_CRATE)



    
    #Update weights
    self.weights += (self.alpha * error * phi)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    
    reward = reward_from_events(self, events)
    
    #Calculate last state's Q(s,a)
    phi = state_to_features(last_game_state,last_action)
    last_q = np.dot(self.weights, phi)

    #Target
    target = reward

    #Error
    error = target - last_q

    #Weight Update
    self.weights += (self.alpha * error * phi)

    #store evaluation data
    # Add final reward
    self.episode_reward += reward

    # Episode statistics
    self.training_rewards.append(
        self.episode_reward
    )

    self.training_lengths.append(
        last_game_state["step"] 
    )


    # Did the agent die?
    died = (
        e.KILLED_SELF in events
        or e.GOT_KILLED in events
    )

    self.training_deaths.append(
        int(died)
    )

    

    results = {
    "rewards": self.training_rewards,
    "episode_lengths": self.training_lengths,
    "deaths": self.training_deaths }

    with open("training_results.pkl", "wb") as file:
        pickle.dump(results, file)

    # Store the model
    with open("actofsafe-saved-model.pt", "wb") as file:
        pickle.dump(self.weights, file)

    # Reset episode statistics
    self.episode_reward = 0
    self.episode_length = 0

    #epsilon decay rate for each episode
    self.epsilon -= 0.0001
    self.epsilon = max(self.epsilon, 0.8)



def reward_from_events(self, events: List[str]) -> float:
    """
    Reward design for custom and game events
    """
    game_rewards = {
        e.SURVIVED_ROUND: 0.1,
        e.COIN_COLLECTED: 30.0,
        e.CRATE_DESTROYED: 0.1,
        e.WAITED: 0,
        e.INVALID_ACTION: -0.5,
        e.KILLED_SELF: -2.0,
        e.GOT_KILLED: -2.0,
        
        #Custom added events
        BOMB_NEXT_TO_CRATE: 0.1,
        BOMB_NOT_NEXT_TO_CRATE: -1.0, 

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
