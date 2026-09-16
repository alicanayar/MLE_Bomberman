import pickle
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features, danger_level

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
#SURVIVED_STEP = "SURVIVED_STEP"
MOVED_INTO_DANGER = "MOVED_INTO_DANGER"
ESCAPED_DANGER = "ESCAPED_DANGER"
STAYED_IN_DANGER = "STAYED_IN_DANGER"
TRAPPED_BY_OWN_BOMB = "TRAPPED_BY_OWN_BOMB"
BOMB_NEXT_TO_CRATE = "BOMB_NEXT_TO_CRATE"
CORRECT_ESCAPE_ACTION = "CORRECT_ESCAPE_ACTION"
REACH_SAFE_TILE = "REACH_SAFE_TILE"
BOMB_NOT_NEXT_TO_CRATE = "BOMB_NOT_NEXT_TO_CRATE"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.alpha = 0.01
    self.gamma = 0.95

    self.epsilon = 1.0

    self.episode_reward = 0
    self.episode_length = 0

    self.training_rewards = []
    self.training_lengths = []
    self.training_deaths = []




def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    
    #to see new actions coordinates
    #old_pos = old_game_state["self"][3]
    #new_pos = new_game_state["self"][3]
#
    #self.logger.info(
    #    f"Step={new_game_state['step']} | "
    #    f"Action={self_action} | "
    #    f"Old={old_pos} | "
    #    f"New={new_pos}"  ) 
    #self.logger.info(f"field:{old_game_state["field"]},field[3,1]:{old_game_state["field"][3,1]}")
    #self.logger.info(f"bombs{new_game_state["bombs"]}")
    
    #Custom event
    #events = list(events)

    #if e.KILLED_SELF not in events and e.GOT_KILLED not in events:
    #    events.append(SURVIVED_STEP)
    
    
    #for action in ACTIONS:
    #    phi = state_to_features(old_game_state, self_action)
    #    print(action, phi)

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

    #custom env 
    if self_action == "BOMB":  #crate_bombing feature
        if phi[-3] == 1:
            events.append(BOMB_NEXT_TO_CRATE)
        else:
            events.append(BOMB_NOT_NEXT_TO_CRATE)

        

    if phi[-2] == 1: #bomb_escape
        events.append(CORRECT_ESCAPE_ACTION)

    if phi[-1] == 1: #safe_after_bomb
        events.append(REACH_SAFE_TILE)


    self.logger.info(f"{self_action}: Q:{current_q:.3f},c_b:{phi[-3]}")
    
    #Update weights
    self.weights += (self.alpha * error * phi)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
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
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.weights, file)

    # Reset episode statistics
    self.episode_reward = 0
    self.episode_length = 0

    #epsilon decay rate for each episode
    self.epsilon -= 0.0003
    self.epsilon = max(self.epsilon, 0.1)


    #q_contributions = self.weights * phi
    #self.logger.info(
    #f"Features={phi}, "
    #f"Weights={self.weights}, "
    #f"Contributions={q_contributions}, "
    #f"Q={last_q}"
#)
    #self.logger.info(f"Updated Weights {self.weights}")


def reward_from_events(self, events: List[str]) -> float:
    """
    
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        #SURVIVED_STEP: +0.01 ,
        e.SURVIVED_ROUND: 0.1, #NORMALLY SETTED 0.1
        e.COIN_COLLECTED: 2.0,
        e.CRATE_DESTROYED: 2.0,
        e.WAITED: 0,
        e.INVALID_ACTION: -0.5,
        e.KILLED_SELF: -2.0,
        e.GOT_KILLED: -2.0,
        BOMB_NEXT_TO_CRATE: 3.0,
        BOMB_NOT_NEXT_TO_CRATE: -1.0, #-0.5 kinda works
        #CORRECT_ESCAPE_ACTION: 0.1,
        #REACH_SAFE_TILE: 0.2
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
