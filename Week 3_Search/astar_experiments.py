import heapq, math
from astar import WAREHOUSE, neighbors, find_pos, reconstruct

def heuristic(state, goal, name):
    dr = abs(state[0] - goal[0])
    dc = abs(state[1] - goal[1])
    if name == "zero":
        return 0
    if name == "manhattan":
        return dr + dc
    if name == "euclidean":
        return math.sqrt(dr * dr + dc * dc)
    if name == "double_manhattan":
        return 2 * (dr + dc)
    raise ValueError(name)

def astar_with_heuristic(grid, name):
    start = find_pos(grid, "S")
    goal = find_pos(grid, "G")
    frontier = [(heuristic(start, goal, name), 0, 0, start)]
    g_cost = {start: 0}
    parent = {start: None}
    action_used = {}
    counter = 0
    expanded = 0

    while frontier:
        f, g, _, current = heapq.heappop(frontier)
        if g != g_cost.get(current):
            continue
        expanded += 1
        if current == goal:
            states, actions = reconstruct(parent, action_used, goal)
            return {"found": True, "length": len(actions), "expanded": expanded}
        for nxt, act in neighbors(grid, current):
            ng = g + 1
            if ng < g_cost.get(nxt, math.inf):
                g_cost[nxt] = ng
                parent[nxt] = current
                action_used[nxt] = act
                counter += 1
                heapq.heappush(
                    frontier,
                    (ng + heuristic(nxt, goal, name), ng, counter, nxt)
                )

    return {"found": False, "length": None, "expanded": expanded}

if __name__ == "__main__":
    for name in ["zero", "manhattan", "euclidean", "double_manhattan"]:
        print(name, astar_with_heuristic(WAREHOUSE, name))
