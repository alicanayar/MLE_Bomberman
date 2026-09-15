import numpy as np
field = np.zeros((17,17))
b = np.ones((17,17))
bombs = [((9,9),0),((3,4),3)]

def danger_level(bombs,field):
    GRID_SIZE = 17
    danger_map = np.zeros(field.shape)
    for (bx,by),t in bombs:
        danger_map[bx,by] = 6-t
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
                    6-t
                )
        
    return danger_map


asd = True
sds = False

if not sds:
    print("aaaa")


