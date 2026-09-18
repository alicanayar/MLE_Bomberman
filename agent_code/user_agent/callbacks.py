import numpy as np
import random
import pickle
from collections import deque


ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
NUM_FEATURES = 11


def setup(self):
    try:
        with open("my-saved-model.pt", "rb") as file:
            self.weights = pickle.load(file)

    except FileNotFoundError:
        self.weights = np.zeros(NUM_FEATURES)

    self.epsilon = 0.1

#write positions after each action (l,r,u,p)
#def position_after_action(action):
    



def get_explosion_zone(bomb_position):
    GRID_SIZE = 17
    explosion_tiles = set()  # Use a set to automatically avoid duplicates

    for bx, by in bomb_position:
        explosion_tiles.add((bx, by))  # Bomb center
        
        for i in range(1, 4):
            # Check 4 directions independently
            for dx, dy in [(i, 0), (-i, 0), (0, i), (0, -i)]:
                nx, ny = bx + dx, by + dy
                
                # Boundary check: ensure coordinates remain inside [0, 16]
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    explosion_tiles.add((nx, ny))
                    
    return list(explosion_tiles) 

      


def BFS(agent_position, coins, field):
    GRID_SIZE = 17

    directions = [
        (0, -1),   # UP
        (0, 1),    # DOWN
        (-1, 0),   # LEFT
        (1, 0)     # RIGHT
    ]

    closest_distance = float('inf')
    closest_path = []
    closest_coin_coord = None

    for coin in coins:

        # Queue contains positions to explore
        queue = deque([agent_position])

        # parent_map[position] = previous position
        parent_map = {
            agent_position: None
        }

        found = False

        while queue:

            current = queue.popleft()
            curr_x, curr_y = current

            # We reached this coin
            if current == coin:
                found = True
                break

            # Explore neighbors
            for dx, dy in directions:

                nx = curr_x + dx
                ny = curr_y + dy

                # Check boundaries
                if not (0 <= nx < GRID_SIZE and
                        0 <= ny < GRID_SIZE):
                    continue

                # Check whether tile is walkable
                if field[ny][nx] != 0:
                    continue

                next_position = (nx, ny)

                # Already visited
                if next_position in parent_map:
                    continue

                parent_map[next_position] = current
                queue.append(next_position)

        # Coin was unreachable
        if not found:
            continue

        # Reconstruct path
        path = []
        current = coin

        while current is not None:
            path.append(current)
            current = parent_map[current]

        # Currently: coin -> ... -> agent
        path.reverse()

        # Number of movements
        distance = len(path) - 1

        # Check if this is the closest coin
        if distance < closest_distance:
            closest_distance = distance
            closest_path = path
            closest_coin_coord = coin

    return closest_distance, closest_path, closest_coin_coord

#Output : distance = 4 path = [(5, 5), (4, 5),(3, 5),(3, 6), (3, 7)]


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




    explosion_zones = get_explosion_zone(bomb_position)

    #MY_FEATURE 1: Bias
    bias = 1.0

    #MY_FEATURE 2: Valid Action
    valid_action = 0.0
    if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
        if field[new_x,new_y] == 0:
            valid_action = 1.0

    #try delete from here
    #elif action == 'BOMB':
    #    if bomb_available:
    #        valid_action = 1.0
    #else: #wait action
    #    valid_action = 1.0

    #MY_FEATURE 4: Current Dangerous check for exploded areas
    is_dangerous = 0.0
    if (x,y) in bomb_position or (x,y) in explosion_zones:
        is_dangerous = 1.0

    #MY_FEATURE 5: Escape Availablity of next step
    

    escape_availability = 0.0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx = new_x + dx
        ny = new_y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx,ny] == 0 and explosion_map[nx,ny] == 0 and (nx,ny) not in explosion_zones and( (nx, ny) not in bomb_position):
                escape_availability += 1
    escape_availability = escape_availability / 4.0
        
    #MY_FEATURE 6: Destination Explosion
    """destination_explosion = 0.0
    explosion_zone = get_explosion_zone(bomb_position)
    if ((new_x,new_y) in explosion_zone or explosion_map[new_x,new_y] > 0) :
        destination_explosion = 1.0"""

    #MY_FEATURE 6 Alternative: Directional Destination Explosion
    #Returns a 4-element list [UP, DOWN, LEFT, RIGHT].
    #Value is 1 if moving in that direction is gonna explode, else 0.
    destination_explosion = [0,0,0,0]
    i = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

        destination_x = x + dx
        destination_y = y + dy

        if 0 <= destination_x < field.shape[0] and 0 <= destination_y < field.shape[1]:

            if ((destination_x, destination_y) in explosion_zones) or (explosion_map[destination_x,destination_y] != 0) or ((destination_x,destination_y) in bomb_position):
                destination_explosion[i] = 1.0

        i += 1


    #FEATURE: shortest walkable path from player to nearest coin via BFS
    #Find the nearest reachable coin from the current state.
    #Determine the shortest path.
    #Check whether action a is the first step of that path.

    #Nearest coin
    #FEATURE: First step toward nearest coin
    _, path,closest_coin_coord = BFS((x,y),coins,field)

    coin_path = 0.0

    if len(path) > 1 and (new_x, new_y) == path[1]:
        coin_path = 1.0

    #FEATURE: Immediate Coin pickup
    coin_pickup = 0.0
    if (new_x,new_y) in coins:
        coin_pickup = 1.0

    #FEATURE: Distance to nearest coin after action
    coin_distance = 0.0
    #j = 0
    #for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    #    if closest_coin_coord is None:
    #        pass
    #    else:
    #        destination_x = x + dx
    #        destination_y = y + dy
#
    #        if 0 <= destination_x < field.shape[0] and 0 <= destination_y < field.shape[1]:
    #            distance = BFS((destination_x,destination_y),[closest_coin_coord],field)[0]
    #            if distance  == float("inf"):
    #                coin_distance[j] = 1.0
    #            else:
    #                coin_distance[j] = distance / 16
    #        j += 1

    #FEATURE above alternative
    if closest_coin_coord is None:
        coin_distance = 1.0
    else:

        distance = BFS(
        (new_x, new_y),
        [closest_coin_coord],
        field
    )[0]

        if distance == float('inf'):
            coin_distance = 1.0
        else:
            coin_distance = distance / 16
    
    # Return as 1D array phi(s, a)
    return np.array([bias,is_dangerous, valid_action, escape_availability, *destination_explosion, coin_path,coin_pickup,coin_distance])



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
