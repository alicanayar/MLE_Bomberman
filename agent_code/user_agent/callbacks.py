import numpy as np
import random
import pickle
from collections import deque


ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
NUM_FEATURES = 12


def setup(self):
    try:
        with open("my-saved-model.pt", "rb") as file:
            self.weights = pickle.load(file)

    except FileNotFoundError:
        self.weights = np.zeros(NUM_FEATURES)

    self.epsilon = 0.0

#write positions after each action (l,r,u,p)
#def position_after_action(action):
    

def get_crates_indices(field):
    """
    Finds all coordinates in a 2D numpy array where the value is -1.
    Returns a list of (x, y) tuples.
    """
    indices = np.argwhere(field == 1)
   
    # Convert numpy array to a list of (x, y) tuples
    return [(x, y) for x, y in indices]

def get_safe_indicies(field):
    indices = np.argwhere(field == 0)
       
        # Convert numpy array to a list of (x, y) tuples
    return [(x, y) for x, y in indices]

def get_explosion_zone(bomb_position):
    GRID_SIZE = 17
    explosion_tiles = set()  # Use a set to automatically avoid duplicates

    for bx, by in bomb_position:
        explosion_tiles.add((bx, by))  # Bomb center
        
        for i in range(1, 4):
            # Check 4 directions independently
            for dx, dy in [(0,-i), (0, i), (-i, 0), (i, 0)]:
                nx, ny = bx + dx, by + dy
                
                # Boundary check: ensure coordinates remain inside [0, 16]
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    explosion_tiles.add((nx, ny))
                    
    return list(explosion_tiles) 


#high number means high danger on the map. Danger levels depends on the bomb timer. High number means bomb gonna explode
# 3 sec = 0.25, 2 sec = 0.50, 1 sec = 0.75, 0 sec = 1.0 (bomb explodes next step)
def danger_level(bombs,field):
    GRID_SIZE = 17
    danger_map = np.zeros(field.shape)
    for (bx,by),t in bombs:
        danger_map[bx,by] =max(danger_map[bx,by] ,(4-t) / 4)
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:

            for i in range(1, 4):

                nx = bx + dx * i
                ny = by + dy * i

                # Outside the map
                if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                    break

                # Wall -> stop this explosion arm
                if field[nx, ny] == -1:
                    break

                # Mark this tile as dangerous
                danger_map[nx, ny] = max(
                    danger_map[nx, ny],
                    (4-t) / 4
                )
        
    return danger_map


def BFS(agent_position, coins, field):
    GRID_SIZE = 17

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

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
                if field[nx][ny] != 0:
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



def BFS_crate(agent_position, field):

    GRID_SIZE = 17

    directions =  [(0, -1), (0, 1), (-1, 0), (1, 0)]

    queue = deque([agent_position])

    parent_map = {
        agent_position: None
    }

    while queue:

        current = queue.popleft()
        x, y = current

        # Check whether current position is next to a crate
        for dx, dy in directions:

            crate_x = x + dx
            crate_y = y + dy

            if 0 <= crate_x < GRID_SIZE and 0 <= crate_y < GRID_SIZE:

                if field[crate_x][crate_y] == 1:

                    # We found a reachable bombing position
                    path = []
                    node = current

                    while node is not None:
                        path.append(node)
                        node = parent_map[node]

                    path.reverse()

                    distance = len(path) - 1

                    return distance, path, (crate_x, crate_y)

        # Explore neighboring walkable tiles
        for dx, dy in directions:

            nx = x + dx
            ny = y + dy

            if not (0 <= nx < GRID_SIZE and
                    0 <= ny < GRID_SIZE):
                continue

            # Only walk on empty tiles
            if field[nx][ny] != 0:
                continue

            next_position = (nx, ny)

            if next_position in parent_map:
                continue

            parent_map[next_position] = current
            queue.append(next_position)

    return float('inf'), [], None

def BFS_safe(agent_position, field, danger_map):

    GRID_SIZE = 17

    directions = [
        (0, -1),   # UP
        (0, 1),    # DOWN
        (-1, 0),   # LEFT
        (1, 0)     # RIGHT
    ]

    queue = deque()

    # State = (position, time)
    queue.append((agent_position, 0))

    # We store the previous state for path reconstruction
    parent_map = {
        (agent_position, 0): None
    }

    while queue:

        current, time = queue.popleft()

        x, y = current

        # ------------------------------------------------
        # Check whether current tile is safe at this time
        # ------------------------------------------------

        danger_value = danger_map[x,y]

        if danger_value == 0:
            # We found a safe destination
            path = []

            state = (current, time)

            while state is not None:
                position, _ = state
                path.append(position)
                state = parent_map[state]

            path.reverse()

            return len(path) - 1, path, current

        # ------------------------------------------------
        # Explore neighboring tiles
        # ------------------------------------------------

        for dx, dy in directions:

            nx = x + dx
            ny = y + dy

            # Boundary check
            if not (0 <= nx < GRID_SIZE and
                    0 <= ny < GRID_SIZE):
                continue

            # Tile must be walkable
            if field[nx,ny] != 0:
                continue

            next_time = time + 1

            # ------------------------------------------------
            # Convert danger value into explosion timing
            #
            # 0.25 -> 3 steps
            # 0.50 -> 2 steps
            # 0.75 -> 1 step
            # 1.00 -> 0 steps
            # ------------------------------------------------

            danger_value = danger_map[nx][ny]

            if danger_value > 0:

                explosion_time = round(
                    4 - danger_value * 4
                )

                # If the tile explodes before or exactly
                # when we arrive, we cannot use it.
                if explosion_time <= next_time:
                    continue

            next_state = ((nx, ny), next_time)

            # Avoid visiting the same position at the same time
            if next_state in parent_map:
                continue

            parent_map[next_state] = (current, time)

            queue.append(next_state)

    # No safe tile found
    return float('inf'), [], None
#Output
#(2, [(1, 1), (1, 2), (1, 3)], (1, 4))

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
        new_y = y - 1

    elif action == 'DOWN':
        new_y = y + 1

    elif action == 'LEFT':
        new_x = x - 1

    elif action == 'RIGHT':
        new_x = x + 1

    elif action == 'BOMB':
        new_x, new_y = x, y

    elif action == 'WAIT':
        new_x, new_y = x, y

    new_x = np.clip(new_x, 0, field.shape[0] - 1)
    new_y = np.clip(new_y, 0, field.shape[1] - 1)
    

    #MY_FEATURE 1: Bias
    bias = 1.0

    #MY_FEATURE 2: Valid Action
    valid_action = 0.0
    if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
        valid_action = int(field[new_x,new_y] == 0)

    elif action == 'BOMB':
        valid_action = int(bomb_available)

    elif action == 'WAIT':
        valid_action = 1

    eval_x = new_x if valid_action else x
    eval_y = new_y if valid_action else y

    dangerous_map = danger_level(bombs,field)

    #THESE TWO FEATURES ARE OPTIONAL: PLAY AROUND IT
    #current_danger = dangerous_map[x,y]
    #current_explosion = int(explosion_map[x,y] > 0)

    #FEATURE: Future danger of destination (how imminent the bomb threat is.)
    destination_danger = dangerous_map[eval_x,eval_y]    
    destination_explosion = int(explosion_map[eval_x,eval_y] > 0)

    is_deadly = float(destination_danger == 1.0 or destination_explosion > 0)

    
    escape_availability = 0.0
    for dx, dy in  [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx = eval_x + dx
        ny = eval_y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx,ny] == 0 and explosion_map[nx,ny] == 0 and  dangerous_map[nx,ny] == 0 :
                escape_availability += 1
    escape_availability = escape_availability / 4.0

    #!!!! long coridor bombs are still problem.

    """        
    _, path,closest_coin_coord = BFS((x,y),coins,field)

    coin_path = 0.0

    if len(path) > 1 and (eval_x, eval_y) == path[1]:
        coin_path = 1.0

    #FEATURE: Immediate Coin pickup
    coin_pickup = 0.0
    if (eval_x,eval_y) in coins:
        coin_pickup = 1.0

    #FEATURE: Distance to nearest coin after action
    coin_distance = 0.0

    if closest_coin_coord is None:
        coin_distance = 1.0
    else:

        distance = BFS(
        (eval_x, eval_y),
        [closest_coin_coord],
        field
    )[0]

        if distance == float('inf'):
            coin_distance = 1.0
        else:
            coin_distance = distance / 16"""
  

    #TRy to give reward for surviving from your own bomb with custom event

     

    
    #Crate explosion
    #A crate that can be hit by a bomb from the agent's current/future position, with a safe escape route available.
    
    #FEATURE: being next to crate
    
    #gives a path closest coordinates next to the crate
    distance, path, crate_coord = BFS_crate((x,y),field) 
    #(2, [(1, 1), (1, 2), (1, 3)], (1, 4))
    

    crate_path = 0.0
    if bomb_available and crate_coord is not None:
        if len(path) > 1 and (eval_x, eval_y) == path[1]:
            crate_path = 1.0

    crate_distance = 0.0
    distance_after_action,_,_ = BFS_crate((eval_x,eval_y),field)
    #if crate_coord != None and len(path) > 1:
    if bomb_available and crate_coord is not None and len(path) > 1:
        crate_distance = 1.0 - (distance_after_action / 16.0)
        #closer to crate →  ~1
        #farther from crate → ~0

    next_to_crate = 0.0
    if crate_coord != None:
        if (eval_x,eval_y) == path[-1]:
            next_to_crate = 1.0
    #agent knows;"I'm next to the crate, but I have an active bomb."
    #next_to_crate = spatial information
    #bomb_available = temporal information

    
    crate_bombing = 0.0
    if bomb_available and next_to_crate == 1.0 and action == "BOMB":
        crate_bombing = 1.0





    #safe_bomb = 0.0
    #if crate_bombing == 1.0 and escape_availability > 0:
    #    safe_bomb = 1.0

    #Safe go
    #Problem:the step that bomb dropped doesn't count


    all_bomb_areas = np.maximum(dangerous_map, explosion_map)
    s_distance, s_path,closest_safe_coord = BFS_safe((x,y),field,all_bomb_areas)
    
    bomb_escape = 0.0
    safe_after_bomb = 0.0

    if not bomb_available:

        if all_bomb_areas[x,y] > 0:
            # We are currently threatened
            if len(s_path) > 1:
                if (eval_x, eval_y) == s_path[1]:
                    bomb_escape = 1.0

        else:
            # Already outside the danger zone
            safe_after_bomb = 1.0


        #if closest_safe_coord is None:
        #    safe_distance = 1.0
        #else:
#
        #    distance = BFS_safe((eval_x, eval_y),field,all_bomb_areas)[0]
#
        #    if distance == float('inf'):
        #        safe_distance = 1.0
        #    else:
        #        safe_distance = distance / 16


   
    
    # Return as 1D array phi(s, a)
    return np.array([bias,valid_action,destination_danger, destination_explosion, is_deadly, escape_availability,
                     #coin_path,coin_pickup,coin_distance, 
                     crate_path, crate_distance,next_to_crate,crate_bombing,bomb_escape,safe_after_bomb])
    
    #return np.array([bias,is_dangerous, valid_action, escape_availability, *destination_explosion, coin_path,coin_pickup,coin_distance,next_to_crate,crate_path,bomb_escape])



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
