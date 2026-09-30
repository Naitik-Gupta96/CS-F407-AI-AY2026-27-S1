import heapq
import math

WAREHOUSE = [
    "#################",
    "#S....#.........#",
    "#.###.#.#######.#",
    "#...#.#.......#.#",
    "###.#.#######.#.#",
    "#...#.........#.#",
    "#.###########.#.#",
    "#.............#G#",
    "#################",
]

MOVES = [
    (-1, 0, "Up"),
    (1, 0, "Down"),
    (0, -1, "Left"),
    (0, 1, "Right"),
]

def find_pos(grid, symbol):
    for r, row in enumerate(grid):
        c = row.find(symbol)
        if c != -1:
            return (r, c)
    return None

def neighbors(grid, state):
    r, c = state
    for dr, dc, action in MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            if grid[nr][nc] != "#":
                yield (nr, nc), action

def manhattan(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def reconstruct(parent, action_used, goal):
    states = [goal]
    actions = []
    current = goal

    while parent[current] is not None:
        actions.append(action_used[current])
        current = parent[current]
        states.append(current)

    states.reverse()
    actions.reverse()
    return states, actions

def astar(grid):
    start = find_pos(grid, "S")
    goal = find_pos(grid, "G")

    if start is None or goal is None:
        return None

    frontier = []
    counter = 0
    g_cost = {start: 0}
    parent = {start: None}
    action_used = {}

    heapq.heappush(
        frontier,
        (manhattan(start, goal), 0, counter, start)
    )

    states_expanded = 0

    while frontier:
        f, current_g, _, current = heapq.heappop(frontier)

        # Ignore stale heap entries.
        if current_g != g_cost.get(current):
            continue

        states_expanded += 1

        if current == goal:
            states, actions = reconstruct(parent, action_used, goal)
            return states, actions, states_expanded

        for nxt, action in neighbors(grid, current):
            new_g = current_g + 1

            if new_g < g_cost.get(nxt, math.inf):
                g_cost[nxt] = new_g
                parent[nxt] = current
                action_used[nxt] = action

                counter += 1
                h = manhattan(nxt, goal)
                f = new_g + h

                heapq.heappush(
                    frontier,
                    (f, new_g, counter, nxt)
                )

    return None

if __name__ == "__main__":
    result = astar(WAREHOUSE)

    if result is None:
        print("No solution found.")
    else:
        states, actions, expanded = result
        print("Solution found")
        print("Actions:", " -> ".join(actions))
        print("Path length:", len(actions))
        print("States expanded:", expanded)
        print("States:", states)
