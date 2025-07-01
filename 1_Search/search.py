
from collections import deque

#BFS vs DFS algorithms for searching a path in a maze

MAZE = [
    [False, True, False, False],
    [False, False, False, True],
    [True, False, False, False],
    [False, False, True, False]
]

START = (0, 0) # top left corner
GOAL = (3, 3) # bottom right corner


#fn to get valid neighbors of a cell
def neighbors(cell, maze):
    """
    Given a cell (r,c) and a 2D maze list, return all adjacent open cells as (r,c) tuples
    """
    r, c = cell
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:   #up, down, left, right
        nr, nc = r + dr, c + dc
        #check bounds and wall
        if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]) and not maze[nr][nc]:
            yield (nr, nc)


#Depth First Search (DFS) algorithm
def dfs(start, goal, maze):
    """
    DFS returns a path from start to goal (if one exists).
    Uses an explicit stack for frontier
    """
    stack = [(start, [start])]  # stack of tuples (cell, path)
    visited = set()  # set of visited cells

    while stack:
        current, path = stack.pop()  # get the last cell and path
        if current == goal:
            return path       #found the goal
        
        if current in visited:
            continue
        visited.add(current)

        #add neigbors to stack. we reverse so that up is explored first...
        for neigbor in neighbors(current, maze):
            if neigbor not in visited:
                stack.append((neigbor, path + [neigbor]))

    return None  # no path found



def bfs(start, goal, maze):
    """
    BFS returns the shortest path (fewest steps)
    from start to goal on an unweighted graph
    """
    #Queue will hold tuples: (current_cell, path_taken)
    queue = deque([(start, [start])])  # queue of tuples (cell, path)
    visited = {start}

    while queue:
        current, path = queue.popleft()  #FIFO - take the oldest entry
        if current == goal:
            return path
        
        for neighbor in neighbors(current, maze):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # no path found




#DEMO
if __name__ == "__main__":
    print("DFS Path:", dfs(START, GOAL, MAZE))
    print("BFS Path:", bfs(START, GOAL, MAZE))