from collections import deque

WAREHOUSE = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################",
]

MOVES = [
    (-1, 0, "Up"),
    (1, 0, "Down"),
    (0, -1, "Left"),
    (0, 1, "Right"),
]


def find_positions(grid):
    start = goal = None

    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "S":
                start = (r, c)
            elif cell == "G":
                goal = (r, c)

    return start, goal


def bfs_path(grid):
    """Find a collision-free path from S to G using breadth-first search."""
    start, goal = find_positions(grid)

    if start is None or goal is None:
        return None

    frontier = deque([start])
    parent = {start: None}
    action_used = {}

    while frontier:
        current = frontier.popleft()

        if current == goal:
            break

        r, c = current

        for dr, dc, action in MOVES:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
                continue

            if grid[nr][nc] == "#":
                continue

            next_state = (nr, nc)

            if next_state in parent:
                continue

            parent[next_state] = current
            action_used[next_state] = action
            frontier.append(next_state)

    if goal not in parent:
        return None

    states = []
    actions = []
    current = goal

    while current != start:
        states.append(current)
        actions.append(action_used[current])
        current = parent[current]

    states.append(start)

    states.reverse()
    actions.reverse()

    return states, actions


def show_result(grid):
    result = bfs_path(grid)

    if result is None:
        print("No collision-free path found.")
        return

    states, actions = result

    print("Path found.")
    print("Actions:")
    print(" -> ".join(actions))
    print("Path length:", len(actions))
    print("States:")
    for i, state in enumerate(states):
        print(f"  {i}: {state}")


if __name__ == "__main__":
    show_result(WAREHOUSE)
