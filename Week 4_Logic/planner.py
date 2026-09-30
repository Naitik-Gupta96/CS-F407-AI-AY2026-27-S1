from collections import deque


def action(name, positive_preconditions=(), negative_preconditions=(),
           positive_effects=(), negative_effects=()):
    return {
        "name": name,
        "pos_pre": set(positive_preconditions),
        "neg_pre": set(negative_preconditions),
        "add": set(positive_effects),
        "delete": set(negative_effects),
    }


def make_move(src, dst):
    return action(
        f"Move({src},{dst})",
        positive_preconditions=[f"At(Robot,{src})"],
        positive_effects=[f"At(Robot,{dst})"],
        negative_effects=[f"At(Robot,{src})"],
    )


def make_pickup(loc):
    return action(
        f"PickUp(Package,{loc})",
        positive_preconditions=[
            f"At(Robot,{loc})",
            f"At(Package,{loc})",
        ],
        negative_preconditions=["Holding(Package)"],
        positive_effects=["Holding(Package)"],
        negative_effects=[f"At(Package,{loc})"],
    )


def make_drop(loc):
    return action(
        f"Drop(Package,{loc})",
        positive_preconditions=[
            f"At(Robot,{loc})",
            "Holding(Package)",
        ],
        positive_effects=[f"At(Package,{loc})"],
        negative_effects=["Holding(Package)"],
    )


def applicable(state, action):
    # S |= Preconditions(action)
    return (
        action["pos_pre"].issubset(state)
        and action["neg_pre"].isdisjoint(state)
    )


def apply_action(state, action):
    # Remove negative effects, then add positive effects.
    next_state = set(state)
    next_state.difference_update(action["delete"])
    next_state.update(action["add"])
    return frozenset(next_state)


def bfs_plan(initial_state, goal, actions):
    """Return a valid action sequence, or None when no plan exists."""
    initial_state = frozenset(initial_state)

    frontier = deque([initial_state])
    parent = {initial_state: None}
    action_used = {}

    while frontier:
        current = frontier.popleft()

        if goal.issubset(current):
            plan = []
            while parent[current] is not None:
                plan.append(action_used[current])
                current = parent[current]
            plan.reverse()
            return plan

        for candidate in actions:
            if not applicable(current, candidate):
                continue

            successor = apply_action(current, candidate)

            if successor in parent:
                continue

            parent[successor] = current
            action_used[successor] = candidate
            frontier.append(successor)

    return None


def execute_plan(initial_state, plan):
    """Execute a plan and return all reached states."""
    state = frozenset(initial_state)
    states = [state]

    for step in plan:
        if not applicable(state, step):
            raise ValueError(f"Invalid action: {step['name']}")
        state = apply_action(state, step)
        states.append(state)

    return states


# -----------------------------
# Original warehouse problem
# -----------------------------
LOCATIONS = ["A", "B", "C"]

ACTIONS = [
    make_move("A", "B"),
    make_move("B", "A"),
    make_move("B", "C"),
    make_move("C", "B"),
]

for loc in LOCATIONS:
    ACTIONS.append(make_pickup(loc))
    ACTIONS.append(make_drop(loc))


INITIAL = {
    "At(Robot,A)",
    "At(Package,A)",
}

GOAL = {
    "At(Package,C)",
}


if __name__ == "__main__":
    plan = bfs_plan(INITIAL, GOAL, ACTIONS)

    if plan is None:
        print("No plan found.")
    else:
        states = execute_plan(INITIAL, plan)

        print("Plan found:")
        for i, step in enumerate(plan, start=1):
            print(f"{i}. {step['name']}")

        print("\nStates:")
        for i, state in enumerate(states):
            print(f"S{i}: {sorted(state)}")

        print("\nGoal achieved:", GOAL.issubset(states[-1]))
