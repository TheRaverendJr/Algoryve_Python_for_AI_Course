from collections import deque


MAZE = [
    [False, True, False, False],
    [False, False, False, True],
    [True, False, False, False],
    [False, False, True, False]
]



START = (0,0) #top left corner
GOAL = (3,3) #bottom right corner

def neighbors(cell, maze):
    # given a cell (r,c)and a 2D maze list, return all adjacent open cells as (r,c) tuples
    r,c = cell
    for dr, dc in [(-1,0), (1,0), (0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        # check bounds and wall
        if (0 <= nr < len(maze) and 0 <= nc < len(maze[0]) and not maze[nr][nc]):
            yield(nr, nc)


#DFS algorithm
def dfs(start, goal, maze):
    stack = [(start, [start])]
    visited = set()  #set of visited cells

    while stack:
        current, path=stack.pop()
        if current == goal:
            return path #succes
        
        if current in visited:
            continue  #skip
        visited.add(current)  #mark as visited

        #add neigbors to stack
        for neighbor in neighbors(current, maze):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None  #no path found




#Breadth First Search Algorithm
def bfs(start, goal, maze):
    queue = deque([(start, [start])]) #FIFO queue
    visited = set()  # set of visited cells

    while queue:
        current, path = queue.popleft()  #remove oldest added cell.
        if current == goal:
            return path
        
        for neighbor in neighbors(current, maze):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None



# Example usage
if __name__ == "__main__":
    print("DFS Path:", dfs(START, GOAL, MAZE))
    print("BFS Path:", bfs(START, GOAL, MAZE))

