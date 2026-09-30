from collections import deque
from astar import find_pos, neighbors, WAREHOUSE, reconstruct

def bfs(grid):
    start = find_pos(grid, "S")
    goal = find_pos(grid, "G")

    if start is None or goal is None:
        return None

    frontier = deque([start])
    parent = {start: None}
    action_used = {}
    states_expanded = 0

    while frontier:
        current = frontier.popleft()
        states_expanded += 1

        if current == goal:
            states, actions = reconstruct(parent, action_used, goal)
            return states, actions, states_expanded

        for nxt, action in neighbors(grid, current):
            if nxt not in parent:
                parent[nxt] = current
                action_used[nxt] = action
                frontier.append(nxt)

    return None

if __name__ == "__main__":
    result = bfs(WAREHOUSE)
    if result is None:
        print("No solution found.")
    else:
        states, actions, expanded = result
        print("Solution found")
        print("Actions:", " -> ".join(actions))
        print("Path length:", len(actions))
        print("States expanded:", expanded)
