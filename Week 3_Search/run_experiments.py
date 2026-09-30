from astar import WAREHOUSE, astar
from bfs import bfs

def run():
    b = bfs(WAREHOUSE)
    a = astar(WAREHOUSE)

    print("BFS:", b[2], "states expanded,", b[1], "path length")
    print("A* :", a[2], "states expanded,", a[1], "path length")

    print("\nHeuristics")
    for name in ["zero", "manhattan", "euclidean", "double_manhattan"]:
        from astar_experiments import astar_with_heuristic
        result = astar_with_heuristic(WAREHOUSE, name)
        print(name, result["length"], result["expanded"])

if __name__ == "__main__":
    run()
