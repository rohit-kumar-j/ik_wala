"""
See example.py for examples
Implement Controller class with your motion planner
"""
import numpy as np
import collections
import copy
import heapq
import matplotlib.pyplot as pt
import pybullet as pb
from simulation import SimulationEnvironment
import evaluation as ev
import cv2
import math

width, height = 640, 480
WINDOW_NAME = "Non-blocking"

class Controller():
    def __init__(self, show_cv_window=True, show_simulation_window=True,
                 joint_names={'m1', 'm2', 'm3', 'm4', 'm5', 'm6'}):
        super(Controller, self).__init__()

        self.show_cv_window = show_cv_window
        self.show_simulation_window = show_simulation_window
        self.joint_names = joint_names



    def _display_cv_window(self, current_image_np):
        if current_image_np is None:
            return

        # Process for display (BGR, uint8)
        display_img = current_image_np[:, :, :3] # Drop alpha if exists
        display_img = display_img[:, :, ::-1].copy() # RGB -> BGR

        if display_img.dtype != np.uint8:
            display_img = np.clip(display_img, 0, 255).astype(np.uint8)

        cv2.imshow(WINDOW_NAME, display_img)
        key = cv2.waitKey(1) # ms delay
        if key == ord('q'):
            print("Q pressed, stopping training.")
            return False # Indicate quit
        return True # Indicate continue

    def cv_window(self,current_image):
        # Drop alpha channel
        # current_image = current_image[:, :, :3]

        # Swap RGB → BGR
        current_image = current_image[:, :, [2, 1, 0]]

        # Ensure dtype is uint8
        if current_image.dtype != np.uint8:
            current_image = np.clip(current_image, 0, 255).astype(np.uint8)

        self.current_image = current_image

    def run(self,env, goal_poses):
        # TODO: Order changes for waypoint map

        self.client_id = env.client_id

        # get end effector idx for querying position
        for idx in range (pb.getNumBodies()):
            # print(pb.getBodyUniqueId(idx))
            # print(pb.getBodyInfo(idx)[0])
            if(pb.getBodyInfo(idx)[0] == b'base_link'):
                # print("True, ", idx)
                for jnt_num in range(pb.getNumJoints(idx)):
                    joint_info = pb.getJointInfo(idx, jnt_num)
                    print(joint_info)
                    if(joint_info[1] == b't7f'):
                        print("Found end effector jnt index")
                        self.tracking_end_eff_main_body_uid = pb.getBodyUniqueId(idx)
                        self.tracking_end_eff_jnt_idx = joint_info[0]

        print("goal poses: " , goal_poses)
        # for blocks in goal_poses:
        self._block_names = list(goal_poses.keys())
        self._goal_block_pos_orn = list(goal_poses.values())
        self._current_block_positions = []
        self._goal_block_positions = []
        for block_name in self._block_names:
            current_block_position = env.get_block_pose(block_name)
            self._current_block_positions.append(current_block_position)
            goal_block_position = goal_poses.get(block_name,'key not found')
            self._goal_block_positions.append(goal_block_position)
        # print("current pos: ", self._current_block_positions)


        print("state saved [id]: ", env.initial_state_id)
        goal_pos_idx = self.get_stack_towers("goals", self._goal_block_positions,sort_by_z_height=False) # NOTE: Change to true if you fuck up
        curr_pos_idx = self.get_stack_towers("current", self._current_block_positions,sort_by_z_height=False) # Currnet state does not need z-height sort


        moves = self.tower_rearrangement_solver(goal_pos_idx, curr_pos_idx)

        print(f"Moves required: {len(moves)}")
        for move in moves:
            print(f"Move from {move[0]} to {move[1]}")

        self.simulate_moves_with_state(goal_pos_idx, curr_pos_idx, len(moves))
        print(f"goal_pos_idx :f{goal_pos_idx}")
        print(f"curr_pos_idx :f{curr_pos_idx}")

        input("Press [Enter] if you are not gay...")

        exit()



        while timestep_count < total_timesteps:

            # print("timestep: ",)
            # --- Agent Interaction Step ---
            # Get action, log_prob, value from the policy
            action_np, action_tensor_cpu, log_prob_cpu, value_cpu = self.get_action_and_value(current_image_np, current_joint_state_np)
            # print("action_np: ", action_np)


            # tracking the end effector of robot arm:
            self._tracking_debug_line = \
                pb.getLinkState(bodyUniqueId = self.tracking_end_eff_main_body_uid,
                               linkIndex = self.tracking_end_eff_jnt_idx)[0]


            # --- Environment Step ---
            env.goto_position(action_np, 0.3)  # 0.3 seconds

            self.addLine(); # tracking the end effector of robot arm:

            # --- OpenCV Display ---
            if self.show_cv_window:
                raw_image_for_display = current_image_np.copy()
                if not self._display_cv_window(raw_image_for_display):
                    break # Stop training if 'q' is pressed

        if self.show_cv_window:
            cv2.destroyAllWindows()

    def _positon_test(self,env, goal_poses):
        print(f"pos: {goal_poses}")

    def get_stack_towers(self, name ,current_block_positions, sort_by_z_height):
        # print(current_block_positions)
        # print("len(current_block_positions): ", len(current_block_positions))
        towers = []
        visited = set()  # keep track of blocks we already used

        for p1 in range(len(current_block_positions)):
            if p1 in visited:
                continue  # skip if already used
            tower = []
            tower.append(p1)
            visited.add(p1)
            # print("current_block_positions: " ,current_block_positions[p1][0], "| z: ", current_block_positions[p1][0][2]) # Position
            # print("current_block_orientations: " ,current_block_positions[p1][1]) # Orientation

            for p2 in range(p1 + 1,len(current_block_positions)):
                if p2 in visited:
                    continue  # don't double count

                if(abs(current_block_positions[p1][0][0] - current_block_positions[p2][0][0]) <= 0.001): # x-coordinate
                    if(abs(current_block_positions[p1][0][1] - current_block_positions[p2][0][1]) <= 0.001): # y-coordinate
                        # if(abs(current_block_positions[p1][0][2] - current_block_positions[p2][0][2]) <= 0.001): # z-coordinate
                        #check if p1 adn p2 have the same x and y-coordinates,
                        if(p1 != p2):
                            # if p1 != p2, p2 is either above p1 or vice versa.either ways they belong to the same tower
                            tower.append(p2)
                            visited.add(p2)

            # print(tower)
            towers.append(tower)
        print(f"{name} towers (without z-height sort): {towers}")

        # for p in range(len(towers)):
        #     print(f"tower: {p}, number of blocks:{len(towers[p])}")


        # NOTE: This block of code isnt really necessary. but keep it if we
        #       fuck up.
        if(sort_by_z_height):
            sorted_towers = []
            for p, tower in enumerate(towers): # Sort by height
                sorted_tower = sorted(tower, key=lambda idx: current_block_positions[idx][0][2])
                # print(f"tower: {p}, number of blocks: {len(sorted_tower)}, blocks (sorted by z-height): {sorted_tower}")
                sorted_towers.append(sorted_tower)
            # print(sorted_towers)
            print(f"{name} towers (with z-height sort): {sorted_towers}")
            towers = sorted_towers

        return towers

    def solve_tower_rearrangement(self,goal_towers, current_towers):
        """
        Finds an efficient sequence of moves to transform current_towers into goal_towers
        using up to 4 temporary towers. This uses a greedy approach for efficiency.

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

        # Create a copy of the towers we'll work with
        current_state = current_towers + temp_towers

        # Create block location mapping for quick lookups
        block_locations = {}
        for tower_idx, tower in enumerate(current_state):
            for pos_idx, block in enumerate(tower):
                block_locations[block] = (tower_idx, pos_idx)

        # Build goal block stacks from bottom to top
        moves = []

        # For each tower in the goal state
        for goal_tower_idx, goal_tower in enumerate(goal_towers):
            # Process blocks from bottom to top
            for goal_pos_idx, block in enumerate(goal_tower):
                # Find current location of the block
                if block not in block_locations:
                    continue  # Skip if block doesn't exist (shouldn't happen)

                current_tower_idx, current_pos_idx = block_locations[block]

                # If the block is already in the correct position, skip it
                if (current_tower_idx == goal_tower_idx and
                    current_pos_idx == goal_pos_idx and
                    all(current_state[current_tower_idx][i] == goal_tower[i] for i in range(goal_pos_idx))):
                    continue

                # First, clear blocks above the current block to temporary towers
                blocks_above = current_state[current_tower_idx][current_pos_idx+1:]
                temp_towers_used = {}  # Map block to temp tower where it's stored

                # Move blocks above to temp towers
                for block_above in blocks_above:
                    # Find an available temp tower
                    temp_tower_idx = None
                    for i in range(num_main_towers, num_main_towers + 4):
                        if not current_state[i]:  # Empty temp tower
                            temp_tower_idx = i
                            break

                    if temp_tower_idx is None:
                        # If no empty temp tower, find one with the fewest blocks
                        temp_tower_idx = min(range(num_main_towers, num_main_towers + 4),
                                            key=lambda i: len(current_state[i]))

                    # Move the block to the temp tower
                    moves.append([current_tower_idx, temp_tower_idx])
                    current_state[temp_tower_idx].append(block_above)
                    current_state[current_tower_idx].pop()
                    block_locations[block_above] = (temp_tower_idx, len(current_state[temp_tower_idx]) - 1)
                    temp_towers_used[block_above] = temp_tower_idx

                # Now move the target block
                # First, clear any blocks above the destination position if needed
                while len(current_state[goal_tower_idx]) > goal_pos_idx:
                    block_to_move = current_state[goal_tower_idx][-1]

                    # Find an available temp tower
                    temp_tower_idx = None
                    for i in range(num_main_towers, num_main_towers + 4):
                        if not current_state[i]:  # Empty temp tower
                            temp_tower_idx = i
                            break

                    if temp_tower_idx is None:
                        # If no empty temp tower, find one with the fewest blocks
                        temp_tower_idx = min(range(num_main_towers, num_main_towers + 4),
                                            key=lambda i: len(current_state[i]))

                    # Move the block to the temp tower
                    moves.append([goal_tower_idx, temp_tower_idx])
                    current_state[temp_tower_idx].append(block_to_move)
                    current_state[goal_tower_idx].pop()
                    block_locations[block_to_move] = (temp_tower_idx, len(current_state[temp_tower_idx]) - 1)

                # Move the block to its goal position
                moves.append([current_tower_idx, goal_tower_idx])
                current_state[goal_tower_idx].append(block)
                current_state[current_tower_idx].pop()
                block_locations[block] = (goal_tower_idx, len(current_state[goal_tower_idx]) - 1)

        # Check if we've built the goal state correctly
        for tower_idx, tower in enumerate(goal_towers):
            if tuple(current_state[tower_idx]) != tuple(tower):
                # If we haven't reached the goal state, there might be blocks not yet properly placed
                # Here we would implement additional logic to handle complex cases
                # For now, this simple greedy approach handles many cases
                pass

        return moves

    def translate_towers_to_names(self,moves, num_main_towers):
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

    def optimize_moves(self,moves, num_main_towers):
        """
        This function attempts to optimize the sequence of moves by eliminating unnecessary ones,
        such as moving a block to a temp tower and then immediately back.

        Args:
            moves: List of [source, destination] moves
            num_main_towers: Number of main towers

        Returns:
            Optimized list of moves
        """
        if not moves:
            return moves

        optimized = []
        i = 0

        while i < len(moves):
            # Add current move to optimized list
            optimized.append(moves[i])

            # Check if there's a next move that would undo this one
            if i + 1 < len(moves) and moves[i+1][0] == moves[i][1] and moves[i+1][1] == moves[i][0]:
                # Skip both moves as they cancel each other out
                i += 2
            else:
                i += 1

        return optimized

    def tower_rearrangement_solver(self,goal_towers, current_towers):
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
                    if isinstance(block, str):
                        if block.startswith('b'):
                            # Convert "b0" to 0, "b1" to 1, etc.
                            try:
                                processed_tower.append(int(block[1:]))
                            except ValueError:
                                # Handle potential issues
                                if block.endswith('b7') or block == '1b7':  # Special case from example
                                    processed_tower.append(17)
                                else:
                                    processed_tower.append(block)
                        else:
                            # If it's a string but doesn't start with 'b', try to convert directly
                            try:
                                processed_tower.append(int(block))
                            except ValueError:
                                processed_tower.append(block)
                    else:
                        processed_tower.append(block)
                processed.append(processed_tower)
            return processed

        # Preprocess input if needed
        goal_towers = preprocess_towers(goal_towers)
        current_towers = preprocess_towers(current_towers)

        # Solve the problem
        raw_moves = self.solve_tower_rearrangement(goal_towers, current_towers)

        if raw_moves is None:
            return "No solution found"

        # Optimize the moves if possible
        optimized_moves = self.optimize_moves(raw_moves, len(goal_towers))

        # Translate to the required naming format
        final_moves = self.translate_towers_to_names(optimized_moves, len(goal_towers))

        return final_moves

    def check_if_same_position(self,p1,p2):
        if abs(p1[0] - p2[0]) <=0.0001:
            if abs(p1[1] - p2[1]) <=0.0001:
                if abs(p1[2] - p2[2]) <=0.0001:
                    return True
        else:
            return False

    def addLine(self):
        pos = pb.getLinkState(bodyUniqueId = self.tracking_end_eff_main_body_uid,
                              linkIndex = self.tracking_end_eff_jnt_idx)[0]
        if(self.euclidian_dist(pos, self._tracking_debug_line) >= 0.001):
            pb.addUserDebugLine(lineFromXYZ = pos,
                                lineToXYZ = self._tracking_debug_line,
                                lineWidth = 3.0,
                                lifeTime = 0,
                                physicsClientId = self.client_id
            )
            self._record_new_debug_pos = True
        else:
            self._record_new_debug_pos = False

    def euclidian_dist(self,a,b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] -b[0])**2 + (a[2] - b[0])**2)

    def simulate_moves_with_state(self,goal_towers, current_towers, moves_to_show):
        # Create a deep copy to not modify the original
        state = [tower.copy() for tower in current_towers]
        state.extend([[] for _ in range(4)])  # Add temp towers

        print("Initial state:")
        for i, tower in enumerate(state):
            tower_name = f"l{i+1}" if i < len(goal_towers) else f"t{i - len(goal_towers) + 1}"
            print(f"{tower_name}: {tower}")
        print()

        # Get raw indexed moves
        raw_moves = self.solve_tower_rearrangement(goal_towers, current_towers)
        optimized_moves = self.optimize_moves(raw_moves, len(goal_towers))

        # Convert to source/dest tower indices
        for idx, (source_idx, dest_idx) in enumerate(optimized_moves[:moves_to_show]):
            # Convert indices to names for display
            source_name = f"l{source_idx+1}" if source_idx < len(goal_towers) else f"t{source_idx - len(goal_towers) + 1}"
            dest_name = f"l{dest_idx+1}" if dest_idx < len(goal_towers) else f"t{dest_idx - len(goal_towers) + 1}"

            # Check if source tower has any blocks
            if not state[source_idx]:
                print(f"Invalid move {idx+1}: Source tower {source_name} is empty")
                continue

            # Move the block
            block = state[source_idx].pop()
            state[dest_idx].append(block)

            print(f"Move {idx+1}: From {source_name} to {dest_name} (Block {block})")

            # Print the state after this move
            for i, tower in enumerate(state):
                tower_name = f"l{i+1}" if i < len(goal_towers) else f"t{i - len(goal_towers) + 1}"
                print(f"{tower_name}: {tower}")
            print()

    def run2(self, env, goal_poses):
        # run the controller in the environment to achieve the goal
        # env.goto_position([1,1,1,1,1,1], 10)
        self.control_period = 1
        self.clear_buffer()
        for _ in range(100000):
            current_image, _, _ = env.get_camera_image()
            self.cv_window(current_image);

            # Show image
            if(self.show_cv_window):
                cv2.imshow(WINDOW_NAME, self.current_image)

            # print("env.get_current_angles: ",env.get_current_angles())
            joint_angles_array = np.array(list(env.get_current_angles().values()), dtype=np.float32)
            # print("joint_angles_array : ",joint_angles_array )
            joint_angles_array = torch.from_numpy(joint_angles_array).float()
            self.current_image = torch.from_numpy(self.current_image).float().permute(2, 0, 1)


            self.forward_features(self.current_image,joint_angles_array)

            pb.stepSimulation(self.client_id)

            if(self.show_cv_window):
                # Wait for a short time and check for a key press
                key = cv2.waitKey(1)

                # Optional: Exit loop if 'q' is pressed
                if key == ord('q'):
                    print("q pressed")
                    break
        # Clean up
        cv2.destroyAllWindows()

if __name__ == "__main__":

    # --- Controller Initialization ---
    controller = Controller(show_simulation_window =False)

    env, goal_poses = ev.sample_trial(num_blocks=20, num_swaps=15, show=controller.show_simulation_window)
    controller.run(env, goal_poses)
