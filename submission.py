"""
See example.py for examples
Implement Controller class with your motion planner
"""
import numpy as np
import collections
from collections import defaultdict
import copy
import heapq
import matplotlib.pyplot as pt
import pybullet as pb
from simulation import SimulationEnvironment
import cv2
import math
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R # Import for orientation calculations


CUBE_SIDE = 0.01905 # .75 inches
BASE_Z = 0.05715 # 2.25 inches
width, height = 640, 480
WINDOW_NAME = "Non-blocking"


class Controller():
    def __init__(self, show_cv_window=True, show_simulation_window=True,
                 joint_names={'m1', 'm2', 'm3', 'm4', 'm5', 'm6'}):

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

    def setup_ik(self, env):
        self.joint_info = env.get_joint_info()
        self.joint_info = list(self.joint_info)

        # Apply offset to 't7f' (assuming it's at index 5 based on your provided joint_info)
        name, parent_idx, translation, orientation, axis = self.joint_info[5]
        # Create new translation tuple with x-offset
        new_translation = (translation[0]-0.010 , translation[1]+0.017 , translation[2]-0.014)
        # Update the joint info with the modified translation
        self.joint_info[5] = (name, parent_idx, new_translation, orientation, axis)

        # Convert back to tuple to maintain immutability if needed
        self.joint_info = tuple(self.joint_info)

        print(f"jotin info: {self.joint_info}")

        # Define the end effector index (we'll use the fixed finger tip, t7f which is at index 5)
        self.end_effector_index = 5

        # Identify the active joints (those with an axis)
        self.active_joints = []
        for i, joint in enumerate(self.joint_info):
            if joint[4] is not None:  # If axis is not None, it's an active joint
                self.active_joints.append(i)


        print(f"n active joints: {self.active_joints}")

        # Joint limits (±90 degrees in radians)
        self.joint_limits = np.array([
                                      [-np.pi/3, np.pi/3], # (jnt_idx:0) is vertical z axis of 1st motor
                                      [-np.pi/2, np.pi/2], # (jnt_idx:1) is horixontal x axis of 2nd motor
                                      [-np.pi/2, np.pi/2], # (jnt_idx:2) is horixontal x axis of 3rd motor

                                      [0,0], # (jnt_idx:3) is vertical z axis of 4th motor
                                      # [-np.pi/2, np.pi/2], # (jnt_idx:3) is vertical z axis of 4th motor

                                      [-np.pi/3, np.pi/3], # (jnt_idx:4) is horixontal x axis of 5th motor
                                      [-np.pi/2, np.pi/2], # (jnt_idx:6) is gripper tip axis of 6th motor
        ])
        print(f"joint_limits: {self.joint_limits}")
        print(f"active joints: {self.active_joints}")

    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

    def forward_kinematics(self, joint_angles):
        """
        Compute the forward kinematics of the robot.

        Args:
            joint_angles: List or array of joint angles in radians for active joints

        Returns:
            End effector position as a numpy array [x, y, z]
        """
        # Initialize transformation matrices
        transforms = [np.eye(4) for _ in range(len(self.joint_info))]

        # Fill in joint angle values for active joints
        angle_idx = 0
        for i in range(len(self.joint_info)):
            if i in self.active_joints:
                # This joint is active, use the provided angle
                angle = joint_angles[angle_idx]
                angle_idx += 1
            else:
                # This joint is fixed, angle is always 0
                angle = 0.0

            # Get joint information
            _, parent_idx, translation, orientation, axis = self.joint_info[i]

            # Create transformation matrix for this joint
            transform = np.eye(4)

            # Apply parent's transformation
            if parent_idx >= 0:
                transform = transforms[parent_idx].copy()

            # Apply translation
            translation_matrix = np.eye(4)
            translation_matrix[:3, 3] = translation
            transform = transform @ translation_matrix

            # Apply orientation
            rotation_matrix = self.quaternion_to_rotation_matrix(orientation)
            orientation_matrix = np.eye(4)
            orientation_matrix[:3, :3] = rotation_matrix
            transform = transform @ orientation_matrix

            # Apply joint rotation if this is an active joint
            if axis is not None:
                # Create rotation matrix around the axis
                c = np.cos(angle)
                s = np.sin(angle)
                x, y, z = axis

                # Rodrigues' rotation formula to get rotation matrix from axis and angle
                cross_product_matrix = np.array([
                    [0, -z, y],
                    [z, 0, -x],
                    [-y, x, 0]
                ])

                R = np.eye(3) * c + (1 - c) * np.outer(axis, axis) + s * cross_product_matrix

                joint_rotation = np.eye(4)
                joint_rotation[:3, :3] = R

                transform = transform @ joint_rotation

            transforms[i] = transform

        # Return the position of the end effector
        return transforms[self.end_effector_index][:3, 3]

    def objective_function(self, joint_angles, target_position):
        """
        Calculate the error between current end effector position and target position.
        This is the function we want to minimize.

        Args:
            joint_angles: Current joint angles
            target_position: Target position for the end effector

        Returns:
            Squared Euclidean distance between current and target positions
        """
        current_position = self.forward_kinematics(joint_angles)
        return np.sum((current_position - target_position) ** 2)

    def inverse_kinematics_fn(self, goal_pose, current_joint_angles_degrees=None,debug_info=False):
        """
        Solves the inverse kinematics problem for a goal pose, with updated coordinate transformation
        to correct the diagonal opposite movement issue.

        Args:
            goal_pose: A tuple with ((x,y,z), (qw,qx,qy,qz)) format
            current_joint_angles_degrees: Current joint angles in degrees
        Returns:
            Dictionary with joint angle values in degrees
        """
        if(debug_info):
            print(f"Processing goal pose: {goal_pose}")

        # Extract position and orientation from the tuple
        original_position = np.array(goal_pose[0])
        orientation = np.array(goal_pose[1])

        # Updated coordinate transformation: flip both x and y
        position = np.array([
            original_position[0],    # Negate x
            original_position[1],    # Negate y
            original_position[2] - BASE_Z      # z stays the same
            # original_position[2] # z stays the same
        ])

        if(debug_info):
            print(f"Original target position: {original_position}")
            print(f"Transformed target position (flipped x and y): {position}")

        # Extract target position (ignore orientation for now)
        target_position = position

        # Define bounds for optimization (joint limits: ±90 degrees)
        bounds = self.joint_limits

        # If current joint angles are provided, use them as initial guess
        # Otherwise, start from zero angles
        if current_joint_angles_degrees is None:
            initial_guess = np.zeros(len(self.active_joints))
        else:
            # Convert the provided angles from degrees to radians
            try:
                # Handle if it's a dictionary
                if isinstance(current_joint_angles_degrees, dict):
                    initial_guess = np.array([math.radians(angle) for angle in current_joint_angles_degrees.values()])
                else:
                    # Handle if it's already an iterable of values
                    initial_guess = np.array([math.radians(angle) for angle in current_joint_angles_degrees])
            except (TypeError, AttributeError):
                # Fallback if conversion fails
                initial_guess = np.zeros(len(self.active_joints))


            if(debug_info):
                print(f"initial guess: {initial_guess}")
            # Ensure initial guess is the right length
            if len(initial_guess) != len(self.active_joints):
                initial_guess = np.zeros(len(self.active_joints))

        # Try multiple initial guesses if needed to avoid local minima
        best_result = None
        best_error = float('inf')

        # List of initial guesses to try (current angles plus some alternatives)
        initial_guesses = [initial_guess]

        # Add some predefined initial configurations that might help avoid local minima
        initial_guesses.append(np.zeros(len(self.active_joints)))  # Zero configuration

        # Try different reasonable starting configurations based on the robot anatomy
        elbow_up = np.zeros(len(self.active_joints))
        elbow_up[1] = np.radians(45)  # Assuming joint 1 is shoulder
        initial_guesses.append(elbow_up)

        elbow_down = np.zeros(len(self.active_joints))
        elbow_down[1] = np.radians(-45)  # Assuming joint 1 is shoulder
        initial_guesses.append(elbow_down)

        # Try each initial guess
        for guess in initial_guesses:
            result = minimize(
                fun=self.objective_function,
                x0=guess,
                args=(target_position,),
                # method='trust-constr',
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-8, 'maxiter': 10000}
                # options={'xtol': 1e-9, 'gtol': 1e-9, 'maxiter': 100000}
            )

            # Calculate error for this result
            error = result.fun

            # Keep the best result
            if error < best_error:
                best_error = error
                best_result = result

        # Get the optimized joint angles in radians from the best result
        joint_angles_rad = best_result.x

        # Convert to degrees
        joint_angles_deg = np.degrees(joint_angles_rad)

        # Clip to ensure within ±90 degree limits
        joint_angles_deg = np.clip(joint_angles_deg, -90, 90)

        # Format the result as a dictionary
        joint_dict = {}
        for i, joint_idx in enumerate(self.active_joints):
            joint_name = self.joint_info[joint_idx][0]
            joint_dict[joint_name] = float(joint_angles_deg[i])

        if(debug_info):
            print(f"Target position error: {best_error}")
            print(f"Joint solution: {joint_dict}")

        # Return the joint dictionary with angles
        return joint_dict

    def add_gripper_offset_to_poses(self, goal_poses_dict):
        CUBE_SIDE = 0.01905 # .75 inches
        dx = CUBE_SIDE / 2.0 + 0.01
        dy = 0 #-(CUBE_SIDE / 2.0) - 0.01
        dz = -0.01

        modified_poses_dict = {}

        for key, pose_value in goal_poses_dict.items():

            original_pos_tuple = pose_value[0]
            original_orient_tuple = pose_value[1] # Keep original orientation

            # Position calculations
            original_x = original_pos_tuple[0]
            original_y = original_pos_tuple[1]
            original_z = original_pos_tuple[2]

            new_x = original_x + dx
            new_y = original_y + dy
            new_z = original_z + dz
            new_pos_tuple = (new_x, new_y, new_z)
            new_pose_value = (new_pos_tuple, original_orient_tuple)

            # Add the modified pose to the new dictionary with the original key
            modified_poses_dict[key] = new_pose_value

        return modified_poses_dict

    def toggle_grip(self,ip,close=False):
        op = ip
        # env.settle(1)
        if(close):
            op["m6"] =-12
        else:
            op["m6"] =0
        # env.goto_position(op,1)
        pb.setJointMotorControl2(
            self.env.robot_id,
            jointIndex = 6,
            controlMode = self.env.control_mode,
            targetPosition = op["m6"],
            targetVelocity = 0,
            positionGain = .25, # important for position accuracy
        )
        # env.settle(2)
        return op

    def calculate_top_positions(self, curr_tower_idx, curr_pos):
        """
        Returns the position of the top of the towers so we can pipe as a
        point into the ik solver
        """
        print(f"\nmain: {curr_tower_idx}")
        for i in range(len(curr_pos)):
            print(f"main pos {i}: {curr_pos[i][0]}")

        arranged_curr_pos=[]
        for i in range(len(curr_tower_idx)):
            print(f"tower{i}, size: {len(curr_tower_idx[i])}")
            tower_pos=[]
            for j in range(len(curr_tower_idx[i])):
                print(f"pos{curr_tower_idx[i][j]}:{curr_pos[curr_tower_idx[i][j]][0]}")
                # print(f"pos:{curr_pos[j][0]}, orn{curr_pos[j][1]}")
                tower_pos.append([f'b{curr_tower_idx[i][j]+1}',curr_pos[curr_tower_idx[i][j]][0]])
            arranged_curr_pos.append(tower_pos)

        # NOTE: arranged_curr_pos is the positions of the towers
        # for i in arranged_curr_pos:
        #     print(i)
        # print(f"arranged_curr_pos:\n {arranged_curr_pos}")

        top_cubes = []
        for i in arranged_curr_pos:
            top_cubes.append(i[-1])
        print(f"top_cubes: {top_cubes}")

        # Give the positions for l1,l2,...; t1,t2,...
        # l1,l2 are towers; t1,t2... are temporary positions
        return top_cubes

    def run(self,env, goal_poses):
        self.client_id = env.client_id
        self.env = env
        pb.changeDynamics(bodyUniqueId= self.env.robot_id,
                          linkIndex = 7,
                          physicsClientId = self.client_id,
                          lateralFriction = 4,
                          spinningFriction = 5,
                          rollingFriction = 2
                          )
        pb.changeDynamics(bodyUniqueId= self.env.robot_id,
                          linkIndex = 8,
                          physicsClientId = self.client_id,
                          lateralFriction = 4,
                          spinningFriction = 5,
                          rollingFriction = 2
                          )

        self.setup_ik(env)

        # get end effector idx for querying position
        for idx in range (pb.getNumBodies()):
            # print(pb.getBodyUniqueId(idx))
            # print(pb.getBodyInfo(idx)[0])
            if(pb.getBodyInfo(idx)[0] == b'base_link'):
                # print("True, ", idx)
                for jnt_num in range(pb.getNumJoints(idx)):
                    joint_info = pb.getJointInfo(idx, jnt_num)
                    # print(joint_info)
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

        # Call this once to get bases of the towers

        # print("state saved [id]: ", env.initial_state_id)
        goal_pos_idx = self.get_stack_towers("goals", self._goal_block_positions,sort_by_z_height=False) # NOTE: Change to true if you fuck up
        curr_pos_idx = self.get_stack_towers("current", self._current_block_positions,sort_by_z_height=False) # Currnet state does not need z-height sort


        self.init_base_pos_of_basic_tower = self._get_tower_bases(curr_pos_idx)
        print("000000000000000000000000000000000000000000000000000000000000000000000000000\n")
        print(f"\nlen: {len(self.init_base_pos_of_basic_tower)} self.init_base_pos_of_basic_tower: {self.init_base_pos_of_basic_tower}")

        # print(f"self.initial_state_tower_bases : {self.initial_state_tower_bases}")
        # print(f"len(self.initial_state_tower_bases) : {len(self.initial_state_tower_bases)}")
        self.temp_towers_bases = [
                        ('t1',((0.12, -0.09, CUBE_SIDE/2),(0.316227766016838, 0.0, 0.0, 0.9486832980505139))),
                        ('t2',((0.09, -0.12, CUBE_SIDE/2),(0.44721359549995804, 0.0, 0.0, 0.8944271909999159))),
                        ('t3',((-0.12,-0.09, CUBE_SIDE/2),(0.9486832980505138, 0.0, 0.0, 0.31622776601683794))),
                        ('t4',((-0.09,-0.12, CUBE_SIDE/2),(0.894427190999916, 0.0, 0.0, 0.447213595499958))),
                      ]
        self.initial_state_tower_bases = self.init_base_pos_of_basic_tower + self.temp_towers_bases
        self.init_base_pos_of_temp_tower = self.temp_towers_bases
        # print(f"self.initial_state_tower_bases : {self.initial_state_tower_bases}")
        # print(f"len(self.initial_state_tower_bases) : {len(self.initial_state_tower_bases)}")

        # NOTE: Add these to the tower solver
        # env._add_block((0.12, -0.09, CUBE_SIDE/2), (0,0,0,1), mass = 2, side=CUBE_SIDE) #t1
        # env._add_block((0.09, -0.12, CUBE_SIDE/2), (0,0,0,1), mass = 2, side=CUBE_SIDE) #t2
        # env._add_block((-0.12,-0.09, CUBE_SIDE/2), (0,0,0,1), mass = 2, side=CUBE_SIDE) #t3
        # env._add_block((-0.09,-0.12, CUBE_SIDE/2), (0,0,0,1), mass = 2, side=CUBE_SIDE) #t4

        moves = self.tower_rearrangement_solver(goal_pos_idx, curr_pos_idx)
        state, optimized_moves = self.prep_optimizer(goal_pos_idx, curr_pos_idx, len(moves))

        #testing
        Wp = []
        for move in moves:
            print(f"Move {move[0]} to {move[1]}")
            # Get source (wp1)
            tower_source = move[0]
            pose = self.get_top_of_tower_pos(state, tower_source, debug_info = False)
            # print(f"tower_src [{tower_source}], pose: {pose}")
            wp1 = (tower_source, pose)

            # Get dest (wp2)
            tower_dest = move[1]
            pose = self.get_top_of_tower_pos(state, tower_dest, debug_info = False)

            #Increase the height of the top tower, so that the waypoint (wp2) is above the tower, and not on it
            # print(f"BEFORE tower_dest [{tower_dest}], pose: {pose}")
            pos = pose[0]
            orn = pose[1]
            pose = ((pos[0], pos[1], pos[2]+CUBE_SIDE),orn)
            # print(f"AFTER tower_dest [{tower_dest}], pose: {pose}")
            # print(f"tower_dest [{tower_dest}], pose: {pose}")
            # print("")
            wp2 = (tower_dest, pose)

            #Modify state after a move so that the positions are consistent

            #get source_idx, dest_idx
            source_idx, dest_idx = self.get_src_and_dest_idx_to_update_state(move,state)
            print(f"source_idx:{source_idx}, dest_idx:{dest_idx} ")

            print(f"Old state: {state}")
            block = state[source_idx].pop()
            print(f"Poping block: {block}")
            state[dest_idx].append(block)
            print(f"Pushing block: {block}")
            print(f"New state: {state}")



            print(f"waypoint 1: {wp1}")
            print(f"waypoint 2: {wp2}")
            print("---------------------------------------------------")
            Wp.append(wp1)
            Wp.append(wp2)
        # print(f"CUBE_SIDEx1: {CUBE_SIDE*1}")
        # print(f"CUBE_SIDEx2: {CUBE_SIDE*2}")
        # print(f"CUBE_SIDEx3: {CUBE_SIDE*3}")
        # print(f"CUBE_SIDEx4: {CUBE_SIDE*4}")
        # print(f"CUBE_SIDEx5: {CUBE_SIDE*5}")

        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print(f"Wp:{Wp}")
        print(f"len(Wp): {len(Wp)}")

        print("---------------------------------------------------")
        print("---------------------------------------------------")


        # See the moves
        print(f"Moves required: {len(moves)}")
        for move in moves:
            print(f"Move from {move[0]} to {move[1]}")

        print("---------------------------------------------------")
        print("---------------------------------------------------")


        simulate_moves = False
        if(simulate_moves):
            self.simulate_moves_with_state(goal_pos_idx, curr_pos_idx, len(moves))
            print(f"goal_pos_idx :f{goal_pos_idx}")
            print(f"curr_pos_idx :f{curr_pos_idx}")
        print("---------------------------------------------------")

        print("---------------------------------------------------")
        print("--------Interweave With intermediate Poses---------")
        print("---------------------------------------------------")

        if(len(Wp)>0):
            Wp = self.insert_intermediate_poses2(Wp)
            goal_poses = self.insert_gripper_alternations(Wp)
        else:
            raise Exception("No moves here")

        # print(f"goal_poses->>>>>>>>>>.: {goal_poses}")
        print(f"goal_poses->>>>>>>>>>.: {goal_poses.keys()}")

        self.op_old = None
        gripper_state = 0.0
        for key, value in goal_poses.items():
            print(f"key: {key}, value: {value}")
            if(key[0]=='c'):
                op = self.op_old
                op = self.toggle_grip(op, True)
                gripper_state = op["m6"]
            elif(key[0]=='o'):
                op = self.op_old
                op = self.toggle_grip(op, False)
                gripper_state = op["m6"]
            if(key[0]=="w" or key[0]=="i"):
                op = self.inverse_kinematics_fn(value ,env.get_current_angles(),debug_info=False)
                self.op_old = op
                op["m6"] = gripper_state
                print(f"op: {op}\n")
            env.goto_position(op,3)
            # self.close_grip()
            # Make a mechanism

    def get_src_and_dest_idx_to_update_state(self, move,state):
        # TESTING:
        # move = ['t3', 'l1']
        # state[-2] = [10]
        # print(f"move: {move}")
        # print(f"state: {state}")
        tower_source = move[0]
        tower_dest = move[1]
        source_idx = int(tower_source[1]) - 1
        dest_idx = int(tower_dest[1]) - 1

        if(tower_dest[0]=="t"):
            dest_idx = dest_idx + (len(state) -4)
        if(tower_source[0]=="t"):
            source_idx = source_idx + (len(state) -4)

        # TESTING:
        # print(f"tower_source:{tower_source[0]}{tower_source[1]}, but source_idx: {source_idx}")
        # print(f"tower_dest:{tower_dest[0]}{tower_dest[1]} , but dest_idx: {dest_idx}")

        # TESTING: Move the block
        # block = state[source_idx].pop()
        # print(f"block: {block}")
        # print(f"state: {state}")
        # state[dest_idx].append(block)
        # print(f"block: {block}")
        # print(f"state: {state}")

        # elif(tower_dest[0]=="l"):
        # exit()

        return source_idx, dest_idx

    def get_top_of_tower_pos(self, state, tower_name, debug_info=True):
        if(debug_info):
            print(f"\nstate: {state}")
        nBasicTowers = len(state) - 4
        if (tower_name[0]=="l"):
            # This is a baisc tower
            tower_idx = int(tower_name[1])
            if(debug_info):
                print(f"tower_idx: {tower_idx}")
            tower_height = len(state[tower_idx - 1])
            if(tower_height>0):
                top_block_num= state[tower_idx - 1][-1]
                top_block_name = f"b{top_block_num+1}"
            else:
                top_block_name = "[empty]"
            if(tower_height>0):
                z_compoment = CUBE_SIDE*tower_height + (CUBE_SIDE/2)
            else:
                if(debug_info):
                    print(f"No twower! Tower height: {tower_height}")
                z_compoment = CUBE_SIDE*0.5 #- CUBE_SIDE
            if(debug_info):
                print(f"Basic tower: height of tower {tower_name} is {tower_height} and it is the {tower_idx}th tower, top_block is {top_block_name}({top_block_num}). The z_comp is CUBE_SIDE*{tower_height}={z_compoment}")
            x = self.init_base_pos_of_basic_tower[tower_idx-1][1][0][0]
            y = self.init_base_pos_of_basic_tower[tower_idx-1][1][0][1]
            orn = self.init_base_pos_of_basic_tower[tower_idx-1][1][1]
            # print(f"orn: {orn}")
            pos = (x,y,z_compoment)
            pose=(pos, orn)
        elif (tower_name[0]=="t"):
            # This is a temporary tower
            tower_idx = int(tower_name[1]) + nBasicTowers
            if(debug_info):
                print(f"tower_idx: {tower_idx}")
            tower_height = len(state[tower_idx-1])
            if(tower_height>0):
                top_block_num= state[tower_idx - 1][-1]
                top_block_name = f"b{top_block_num+1}"
            else:
                top_block_name = "e"
            if(debug_info):
                print(f"Temp tower: height of tower {tower_name} is {tower_height} and it is the {tower_idx}th tower, top_block is [{top_block_name}mpty]")
            if(tower_height>0):
                z_compoment = CUBE_SIDE*tower_height + (CUBE_SIDE/2)
            else:
                z_compoment = CUBE_SIDE*0.5 #- CUBE_SIDE
            x = self.init_base_pos_of_temp_tower[int(tower_name[1])-1][1][0][0]
            y = self.init_base_pos_of_temp_tower[int(tower_name[1])-1][1][0][1]
            # orn = self.init_base_orn_of_basic_tower[tower_idx-1][1][1]
            orn = self.init_base_pos_of_temp_tower[int(tower_name[1])-1][1][1]
            pos = (x,y,z_compoment)
            pose=(pos, orn)
        else:
            raise Exception(f"Not a valid tower {tower_name}")
        return pose

    def get_wps(self,state):
        wp1 = []
        i1 = []
        i2 = []
        wp2 = []
        for i in range(len(state)):
            for j in state[i]:
                print(f"i,j: {i, j}")
                pose = env.get_block_pose(f'b{j+1}')
                print(f"pose of block b{j+1}: {pose}")
                i_pos = (pose[0][0],pose[0][1],0.2)
                i_quat = pose[1]
                i1 = (i_pos, i_quat)
                print(f"pose of i_1 : {i1}")
        exit()

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

    def prep_optimizer(self, goal_towers, current_towers, moves_to_show):
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

        return state, optimized_moves

    def _get_tower_bases(self, state):
        """
            Returns the list of the cubes which are on bottom of towers

            NOTE: This is to be called at the beginning so that we can get the x,y
            positions of the tower bases
        """
        # print(f"state >>> : {state}")
        tower_bases = []
        for tower in range(len(state)):
            if(len(state[tower])>0):
                # print(f"tower >>> : {state[tower][0]}")
                block_name = f'b{state[tower][0]+1}'
                # print(f"tower {block_name} pos: {self.env.get_block_pose(block_name)}")
                tower_bases.append((block_name, self.env.get_block_pose(block_name)))
            else:
                # print(f"Empty tower {tower+1} ...... adding from intial_state to e{tower+1}: {self.initial_state_tower_bases[tower]}")
                pos = self.initial_state_tower_bases[tower][1][0]
                # print(f"pos: {pos}")
                orn = self.initial_state_tower_bases[tower][1][1]
                # print(f"orn: {orn}")
                tower_bases.append((f'e{tower+1}', (pos,orn)))

        return tower_bases

    def get_top_cubes(self, state):
        """ Returns the list of the cubes which are on top of towers"""
        # NOTE: Add these to the tower solver
        # env._add_block((0.12, -0.09, CUBE_SIDE/2), (0,0,0,1), mass = 2, side=CUBE_SIDE) #t1
        # env._add_block((0.09, -0.12, CUBE_SIDE/2), (0,0,0,1), mass = 2, side=CUBE_SIDE) #t2
        # env._add_block((-0.12,-0.09, CUBE_SIDE/2), (0,0,0,1), mass = 2, side=CUBE_SIDE) #t3
        # env._add_block((-0.09,-0.12, CUBE_SIDE/2), (0,0,0,1), mass = 2, side=CUBE_SIDE) #t4
        # print(f"state#############: {state}")
        bottom_of_current_towers = self._get_tower_bases(state)
        # This is the original state: self.initial_state_tower_bases
        # print(f"bottom_of_current_towers : {bottom_of_current_towers }")

        pose_state_of_blocks =[]
        for tower in state:
            block_pose = []
            for block in tower:
                block_name = f"b{block+1}"
                if(False):# Only Debug Print
                    print(f"block No.:{block}, block_name={block_name} pose:{self.env.get_block_pose(block_name)}")
                block_pose.append((block_name ,self.env.get_block_pose(block_name)))
            pose_state_of_blocks.append(block_pose)
        # print(f"pose_state_of_blocks: {pose_state_of_blocks}")

        top_cubes = []
        for tower in range(len(pose_state_of_blocks)):
            try:
                # print(f"tower: {pose_state_of_blocks[tower][-1]}")
                top_cubes.append(pose_state_of_blocks[tower][-1])
            except:
                # print(f"Tower {tower + 1} is empty, skipping............")
                top_cubes.append((bottom_of_current_towers[tower])) # Add empty if there is no tower # TODO: Change this to be able to add empty tower base positions
        # print(f"top_cubes: {top_cubes}")
        return top_cubes

    def simulate_moves_with_state(self,goal_towers, current_towers, moves_to_show):

        print(f"moves_to_show: {moves_to_show}")

        state, optimized_moves = self.prep_optimizer(goal_towers, current_towers, moves_to_show )

        print(f"optimized_moves: {optimized_moves}")

        # Convert to source/dest tower indices
        for idx, (source_idx, dest_idx) in enumerate(optimized_moves[:moves_to_show]):
            print(f"start state:{state}")
            # print(f"special: idx={idx}, source_idx={source_idx}, dest_idx={dest_idx}")
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

            top_cubes = self.get_top_cubes(state)
            # print(f"top_cubes: {top_cubes}")

            print(f"Move {idx+1}: From {source_name} to {dest_name} (Block {block})")

            # Print the state after this move
            print("state after move:")
            for i, tower in enumerate(state):
                tower_name = f"l{i+1}" if i < len(goal_towers) else f"t{i - len(goal_towers) + 1}"
                print(f"{tower_name}: {tower}")
            print()

    def insert_intermediate_poses2(self, waypoints_list):
        expanded_waypoints = []

        # Process all waypoints except the last one
        for i in range(len(waypoints_list)-1):
            # Current waypoint
            current_wp = waypoints_list[i]
            current_pos = current_wp[1][0]  # (x,y,z)
            current_orn = current_wp[1][1]  # (w,x,y,z)

            # Next waypoint
            next_wp = waypoints_list[i+1]
            next_pos = next_wp[1][0]  # (x,y,z)
            next_orn = next_wp[1][1]  # (w,x,y,z)

            # Add current waypoint to result
            expanded_waypoints.append(current_wp)

            # Add first intermediate point (above current waypoint)
            i1_pos = (current_pos[0], current_pos[1], 0.2)  # Same x,y as current, z=0.2
            expanded_waypoints.append(('i_temp', (i1_pos, current_orn)))

            # Add second intermediate point (above next waypoint)
            i2_pos = (next_pos[0], next_pos[1], 0.2)  # Same x,y as next, z=0.2
            expanded_waypoints.append(('i_temp', (i2_pos, next_orn)))

        # Add the last waypoint
        expanded_waypoints.append(waypoints_list[-1])

        # Step 2: Convert to sequential dictionary
        sequential_dict = {}

        # Track original waypoints and intermediate points separately
        wp_counter = 1
        i_counter = 1

        for _, pose in expanded_waypoints:
            if pose[0][2] == 0.2:  # This is an intermediate point (z=0.2)
                key = f'i{i_counter}'
                i_counter += 1
            else:  # This is an original waypoint
                key = f'w{wp_counter}'
                wp_counter += 1

            sequential_dict[key] = pose
        print("[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]")
        print("[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]")
        print(sequential_dict)
        print("[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]")
        print("[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]")

        for key, value in sequential_dict.items():
            print(f"key: {key}")
        return sequential_dict

        # exit()

    def insert_gripper_alternations(self, waypoints_dict):
        ordered_items = list(waypoints_dict.items())
        result_items = []

        grip_state = 'c'  # Start with close gripper
        grip_counter = 1

        # Process each item in order
        i = 0
        while i < len(ordered_items):
            key, value = ordered_items[i]

            # Add the original item
            result_items.append((key, value))

            # If this is a waypoint, add grip command right after it
            if key.startswith('w'):
                if grip_state == 'c':
                    grip_key = f'c{grip_counter}'
                    result_items.append((grip_key, 'grip'))
                    grip_state = 'o'
                else:
                    grip_key = f'o{grip_counter}'
                    result_items.append((grip_key, 'grip'))
                    grip_state = 'c'
                    grip_counter += 1

            i += 1

        result_dict = dict(result_items)

        # print(result_dict)
        # for key, value in result_dict.items():
        #     print(key)
        # Convert back to ordered dictionary
        return dict(result_items)

    def insert_intermediate_poses(self, goal_poses):
        """
        Insert intermediate poses between each pair of blocks in goal_poses.

        For each pair (b1, b2), (b2, b3), etc., insert (i1, i2), (i3, i4), etc.
        where i1 has the same x,y as b1 but z=0.2, and i2 has the same x,y as b2 but z=0.2.

        Args:
            goal_poses: Dictionary of block poses {block_name: ((x, y, z), (qx, qy, qz, qw))}

        Returns:
            Dictionary with intermediate poses inserted
        """
        # Sort block names by number
        # block_names = sorted(goal_poses.keys(), key=lambda x: int(x[1:]))
        # print(f"block_names: {block_names}")
        block_names = list(goal_poses.keys())

        # Create new dictionary to store results
        new_poses = {}

        gripper_state = "close" # False is open, True is closed
        # Process all blocks except the last one
        for i in range(len(block_names) - 1):
            current_block = block_names[i]
            next_block = block_names[i+1]

            # Add the current block
            new_poses[current_block] = goal_poses[current_block]

            # Get positions and orientations
            current_pos, current_orient = goal_poses[current_block]
            next_pos, next_orient = goal_poses[next_block]

            # Create intermediate poses with z=0.2
            i1_name = f"i{2*i+1}" # Move to intermediate pt 1
            i2_name = f"i{2*i+2}" # Move to intermediate pt 2

            # NOTE: Sequence:
            # # ... Moves to b1, where there is a block
            # g_name = "c" # Grip the block
            # i1_name = f"i{2*i+1}" # Move to intermediate pt 1
            # i2_name = f"i{2*i+2}" # Move to intermediate pt 2
            # # ... Moves to b2, where there is a block
            # g_name = "o" # Ungrip the block at pt 2
            # # ... Moves to i3
            # # ... Moves to i4
            # # ... Moves to b3, where there is a block

            # i1 has same x,y as current_block but z=0.2, same orientation
            i1_pos = (current_pos[0], current_pos[1], 0.2)
            i1_orient = current_orient

            # i2 has same x,y as next_block but z=0.2, same orientation
            i2_pos = (next_pos[0], next_pos[1], 0.2)
            i2_orient = next_orient

            # Add intermediate poses
            new_poses[i1_name] = (i1_pos, i1_orient)
            new_poses[i2_name] = (i2_pos, i2_orient)

        # Add the last block
        new_poses[block_names[-1]] = goal_poses[block_names[-1]]

        print(new_poses)
        print(new_poses.keys())
        original_b_names = [name for name in block_names if name.startswith('b')]

        # Create the new list
        modified_block_names = []
        grip_state = "o"

        final_dict = {}

        i=1
        for key in new_poses.keys():
            # print(f"key:{key}")
            if(key[0]=="b"):
                modified_block_names.append(key)
                if(grip_state=="c"):
                    modified_block_names.append(f'o{i}')
                    grip_state = 'o'
                else:
                    modified_block_names.append(f'c{i}')
                    grip_state = 'c'
                # print("add grip")
            else: # key == i
                modified_block_names.append(key)
            i+=1
        modified_block_names.append('k')
        # print(f"modified_block_names; {modified_block_names}")

        for key in modified_block_names:
            if(key[0]=='b' or key[0]=='i'):
                # print(key)
                final_dict[key] = new_poses[key]
            else:
                # print("c/o/k")
                final_dict[key] = "gripper"

        return final_dict

if __name__ == "__main__":

    import evaluation as ev
    controller = Controller(show_simulation_window = True)

    env, goal_poses = ev.sample_trial(num_blocks=10, num_swaps=4, show=controller.show_simulation_window)
    print(f"goal_poses:{goal_poses}")

    BASE_Z = 0.05715 # 2.25 inches
    CUBE_SIDE = 0.01905 # .75 inches

    controller.run(env, goal_poses)

    accuracy, loc_errors, rot_errors = evaluate(env, goal_poses)

    # env.close()

    print(f"\n{int(100*accuracy)}% of blocks near correct goal positions")
    print(f"mean|max location error = {np.mean(loc_errors):.3f}|{np.max(loc_errors):.3f}")
    print(f"mean|max rotation error = {np.mean(rot_errors):.3f}|{np.max(rot_errors):.3f}")
