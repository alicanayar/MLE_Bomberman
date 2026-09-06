import numpy as np

grid= np.array([
        [ -1,  0,  1 ],
        [  4,  10,  8 ],
        [ -5,  4, -1 ]
    ])
x=2
y=1
me = (x,y)
# print(x[me[y]][me[x]])
#print(grid[y,x])





def trial(prime):
     # --- FEATURE 1: Does action 'a' step closer to nearest coin? ---
        closer_to_coin = 0.0
        if coins:
            old_distances = [abs(cx - x) + abs(cy - y) for cx, cy in coins]
            new_distances = [abs(cx - new_x) + abs(cy - new_y) for cx, cy in coins]
            if min(new_distances) < min(old_distances):
                closer_to_coin = 1.0
    
        # --- FEATURE 2: Does action 'a' hit a wall or crate? ---
        hits_obstacle = 0.0
        if field[new_x, new_y] != 0: # 0 means free tile
            hits_obstacle = 1.0
    
        # --- FEATURE 3: Is action 'a' dropping a bomb near a crate? ---
        bombs_crate = 0.0
        if action == 'BOMB':
            # Check neighboring tiles for crates
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            if any(field[nx, ny] == 1 for nx, ny in neighbors):
                bombs_crate = 1.0
        
        return closer_to_coin

bombs = [
    ((1, 3), 3),
    ((4, 5), 0),
    ((7, 2), 0)
]
            
def expl(bombs):
    # Keep only tuples where countdown (t) equals 0
    # Keep only bomb's coordinates where countdown (t) is 0 
    bomb_position = [coords for coords, t in bombs if t == 0]
    explosion_tiles = []
    for bomb in bomb_position:
        explosion_tiles.append(bomb)
        for i in range(1,3):
            explosion_tiles.append((bomb[0]+i,bomb[1]))
            explosion_tiles.append((bomb[0]-i,bomb[1]))
            explosion_tiles.append((bomb[0],bomb[1]+i))
            explosion_tiles.append((bomb[0],bomb[1]-i))

    
    return explosion_tiles  



asds= np.array([1,2,3,4,5,6])
weights = np.zeros(5)
print(weights)