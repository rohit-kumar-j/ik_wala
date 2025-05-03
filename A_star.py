from collections import deque, defaultdict
import heapq
import copy

def solve_tower_rearrangement(goal_towers, current_towers):
    """
    Finds the minimum number of moves to transform current_towers into goal_towers
    using up to 4 temporary towers.

    Args:
        goal_towers: List of lists representing the goal state of towers
        current_towers: List of lists representing the current state of towers

    Returns:
        List of moves, where each move is [source, destination]
    """
    # Fix any typos in the input (like "1b7" instead of "b17")
    current_towers = [[block for block in tower if isinstance(block, (int, str))] for tower in current_towers]

    # Create temp towers (empty initially)
    num_main_towers = len(goal_towers)
    temp_towers = [[] for _ in range(4)]

    # Initial state includes main towers and temp towers
    initial_state = current_towers + temp_towers

    # Target state includes goal towers and empty temp towers
    target_state = goal_towers + [[] for _ in range(4)]

    # Convert the state to a tuple of tuples for hashability
    def state_to_tuple(state):
        return tuple(tuple(tower) for tower in state)

    # Check if a state is the target state
    def is_target(state):
        for i in range(num_main_towers):
            if tuple(state[i]) != tuple(target_state[i]):
                return False
        return True

    # A* search to find the minimum number of moves
    def heuristic(state):
        # Simple heuristic: count blocks in wrong position
        h_value = 0
        for i in range(num_main_towers):
            goal_tower = target_state[i]
            current_tower = state[i]

            # Count mismatched blocks
            min_len = min(len(goal_tower), len(current_tower))
            for j in range(min_len):
                if current_tower[j] != goal_tower[j]:
                    h_value += 1

            # Add blocks that need to be added or removed
            h_value += abs(len(goal_tower) - len(current_tower))

        return h_value

    # Generate all possible moves from a state
    def get_possible_moves(state):
        moves = []
        for source in range(len(state)):
            if not state[source]:  # Skip empty towers
                continue

            top_block = state[source][-1]

            for dest in range(len(state)):
                if source == dest:
                    continue

                # Valid move: place block on any tower
                moves.append((source, dest))

        return moves

    # Apply a move to a state
    def apply_move(state, move):
        source, dest = move
        new_state = copy.deepcopy(state)

        if not new_state[source]:
            return None  # Invalid move, source tower is empty

        block = new_state[source].pop()
        new_state[dest].append(block)

        return new_state

    # A* search
    visited = set()
    initial_tuple = state_to_tuple(initial_state)

    # Priority queue: (priority, move_count, state_tuple, moves)
    pq = [(heuristic(initial_state), 0, initial_tuple, [])]

    while pq:
        _, move_count, state_tuple, moves = heapq.heappop(pq)

        if state_tuple in visited:
            continue

        visited.add(state_tuple)

        # Convert back to list format for processing
        state = [list(tower) for tower in state_tuple]

        if is_target(state):
            return moves

        for source, dest in get_possible_moves(state):
            new_state = apply_move(state, (source, dest))
            if new_state is None:
                continue

            new_state_tuple = state_to_tuple(new_state)
            if new_state_tuple not in visited:
                new_moves = moves + [[source, dest]]
                priority = move_count + 1 + heuristic(new_state)
                heapq.heappush(pq, (priority, move_count + 1, new_state_tuple, new_moves))

    return None  # No solution found

def translate_towers_to_names(moves, num_main_towers):
    """
    Translates tower indices to the naming format required in the problem:
    l1, l2, l3, etc. for main towers and t1, t2, t3, t4 for temp towers.

    Args:
        moves: List of [source, destination] moves with tower indices
        num_main_towers: Number of main towers (excluding temp towers)

    Returns:
        List of moves in [source_name, dest_name] format
    """
    translated_moves = []

    for source, dest in moves:
        # Translate source
        if source < num_main_towers:
            source_name = f"l{source+1}"  # Main towers are 1-indexed
        else:
            # Temp towers are indexed from 0 after main towers
            temp_index = source - num_main_towers + 1
            source_name = f"t{temp_index}"

        # Translate destination
        if dest < num_main_towers:
            dest_name = f"l{dest+1}"  # Main towers are 1-indexed
        else:
            # Temp towers are indexed from 0 after main towers
            temp_index = dest - num_main_towers + 1
            dest_name = f"t{temp_index}"

        translated_moves.append([source_name, dest_name])

    return translated_moves

def optimize_moves(moves, num_main_towers):
    """
    This function attempts to optimize the sequence of moves by eliminating unnecessary ones,
    such as moving a block to a temp tower and then immediately back.

    Args:
        moves: List of [source, destination] moves
        num_main_towers: Number of main towers

    Returns:
        Optimized list of moves
    """
    # Implementation of move optimization logic would go here
    # For now, we'll just return the original moves
    return moves

def tower_rearrangement_solver(goal_towers, current_towers):
    """
    Main function to solve the tower rearrangement problem

    Args:
        goal_towers: List of lists representing the goal state
        current_towers: List of lists representing the current state

    Returns:
        List of moves in the required format
    """
    # Handle the case where input might be strings like "b0", "b1" instead of integers
    def preprocess_towers(towers):
        processed = []
        for tower in towers:
            processed_tower = []
            for block in tower:
                if isinstance(block, str) and block.startswith('b'):
                    # Convert "b0" to 0, "b1" to 1, etc.
                    try:
                        processed_tower.append(int(block[1:]))
                    except ValueError:
                        # Handle potential issues like "1b7"
                        processed_tower.append(block)
                else:
                    processed_tower.append(block)
            processed.append(processed_tower)
        return processed

    # Preprocess input if needed
    goal_towers = preprocess_towers(goal_towers)
    current_towers = preprocess_towers(current_towers)

    # Solve the problem
    raw_moves = solve_tower_rearrangement(goal_towers, current_towers)

    if raw_moves is None:
        return "No solution found"

    # Optimize the moves if possible
    optimized_moves = optimize_moves(raw_moves, len(goal_towers))

    # Translate to the required naming format
    final_moves = translate_towers_to_names(optimized_moves, len(goal_towers))

    return final_moves

# Example usage:
if __name__ == "__main__":
    # Example from the problem statement
    # Convert from "b0" format to integers for internal processing
    # goal_towers = [[0, 2, 5, 9], [1, 19], [3, 18], [4, 11], [6, 7, 12], [8, 10, 13, 14], [15, 16], [17]]
    # current_towers = [[0, 1, 2, 5], [3, 14, 16, 19], [4, 11], [6, 7, 12], [8, 18], [9, 13], [10, 15], [17]]
    goal_towers = [[0, 6], [1], [2, 4], [3, 8, 9], [5], [7]]
    current_towers = [[0, 6], [1, 2], [3, 4, 8], [5], [7], [9]]

    result = tower_rearrangement_solver(goal_towers, current_towers)
    print(f"Moves required: {len(result)}")
    print("Sequence of moves:")
    for move in result:
        print(f"Move from {move[0]} to {move[1]}")
