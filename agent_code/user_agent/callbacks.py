import numpy as np
import random
import pickle
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
NUM_FEATURES = 5


def setup(self):
    try:
        with open("my-saved-model.pt", "rb") as file:
            self.weights = pickle.load(file)

    except FileNotFoundError:
        self.weights = np.zeros(NUM_FEATURES)

    
    #self.weights = np.zeros(NUM_FEATURES) 
    self.epsilon = 0.1

def get_explosion_zone(bomb_position):
    # Keep only tuples where countdown (t) equals 0
    # Keep only bomb's coordinates where countdown (t) is 0 

    explosion_tiles = []
    for bomb in bomb_position:
        explosion_tiles.append(bomb)
        for i in range(1,4):
            explosion_tiles.append((bomb[0]+i,bomb[1]))
            explosion_tiles.append((bomb[0]-i,bomb[1]))
            explosion_tiles.append((bomb[0],bomb[1]+i))
            explosion_tiles.append((bomb[0],bomb[1]-i))

    
    return explosion_tiles  

def state_to_features(game_state: dict, action: str) -> np.ndarray:
    """
    Constructs feature vector phi(s, a) from raw game_state for a specific action.
    """
    if game_state is None:
        return np.zeros(NUM_FEATURES) # Return zero vector if state doesn't exist

    # --- Extract raw variables from game_state ---
    field = game_state["field"] # (WxH)   (1) crates, (-1) walls, (0) free tiles
    bombs = game_state['bombs']
    bomb_position = [xy for (xy, t) in bombs] # (x1,y1), (x2,y2) bomb coordinates
    bomb_timer = [t for (xy, t) in bombs] # (t1), (t2) bomb timer for each bomb
    explosion_map = game_state["explosion_map"] # (WxH)  (0) safe, else dangerous
    coins = game_state["coins"] #(x,y) coins coordinates
    _, score, bomb_available, (x, y) = game_state['self'] # n, score, have bomb or not, (x,y) agent coordinates
    enemy_position = [xy for (_, _, _, xy) in game_state['others']] # (x1,y1), (x2,y2), (x3,y3) agent coordinates


    # Simulate where action 'a' takes us
    new_x, new_y = x, y
    if action == 'UP':
        if y == 0:
            new_y = y
        else:
            new_y -= 1
        
    elif action == 'DOWN':
        if y == 16:
            new_y = y
        else:
            new_y += 1

    elif action == 'LEFT':
        if x == 0:
            new_x = x
        else:
            new_x -= 1
    elif action == 'RIGHT': 
        if x == 16:
            new_x = x
        else:
            new_x += 1

    else : new_x, new_y = x, y 

    #MY_FEATURE 1: Bias
    bias = 1.0
    #MY_FEATURE 2: Valid Action
    valid_action = 0.0
    if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
        if field[new_x,new_y] == 0:
            valid_action = 1.0
    elif action == 'BOMB':
        if bomb_available:
            valid_action = 1.0
    else: #wait action
        valid_action = 1.0

    #MY_FEATURE 3: Player position movement for left, right, up, down
    #hits_obstacle = 0.0
    #if field[new_x,new_y] in (-1,1):
    #    hits_obstacle = 1.0


    #MY_FEATURE 4: Dangerous Neighbors
    is_dangerous = 0.0
    if explosion_map[new_x,new_y] != 0:
        is_dangerous = 1.0

    #MY_FEATURE 5: Escape Availablity
    escape_available = 0.0
    if explosion_map[new_x,new_y] == 0 and field[new_x,new_y] == 0:
        if action in ['UP', 'DOWN', 'LEFT', 'RIGHT'] and (new_y == y or new_x == x):
            escape_available = 0.0
        else:
            escape_available = 1.0


    #MY_FEATURE 6: Destination Explosion
    destination_explosion = 0.0
    explosion_zone = get_explosion_zone(bomb_position)
    if ((new_x,new_y) in explosion_zone or explosion_map[new_x,new_y] > 0) :
        destination_explosion = 1.0

    # Return as 1D array phi(s, a)
    return np.array([bias, valid_action, is_dangerous, escape_available, destination_explosion])



def act(self, game_state: dict) -> str:
    # 1. Compute Q(s, a) for all actions
    q_values = []
    for action in ACTIONS:
        # Generate feature vector for this (state, action) pair
        phi_sa = state_to_features(game_state, action)
        
        # Q(s, a) = dot product of w and phi(s, a)
        q_val = np.dot(self.weights, phi_sa)
        q_values.append(q_val)
    
    # 2. choosing the action using epsilon-greedy
    if random.random() < self.epsilon:
        action = random.choice(ACTIONS)
    else:
        best_action_index = np.argmax(q_values)
        action =  ACTIONS[best_action_index]

    
    return action
