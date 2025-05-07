import itertools as it
import numpy as np
import matplotlib.pyplot as pt
import pybullet as pb
from simulation import SimulationEnvironment, _pb_to_quat
from submission import Controller

CUBE_SIDE = 0.01905 # .75 inches

def get_tower_base_poses(half_spots=4):
    radii = (-.14, -.19)
    alpha = 0.5*1.57
    thetas = (3.14 - alpha)/2 + alpha * np.arange(half_spots) / (half_spots-1)

    bases = ()
    for (r, theta) in it.product(radii, thetas):
        pos = (r*np.cos(theta), r*np.sin(theta), -0.5*CUBE_SIDE)
        quat = _pb_to_quat(pb.getQuaternionFromEuler((0,0,theta)))
        bases += ((pos, quat),)

    return bases

def sample_trial(num_blocks, num_swaps, show=True):
    env = SimulationEnvironment(show=show)

    # sample initial block positions
    bases = get_tower_base_poses()

    # stack some random blocks
    towers = [[base] for base in bases]
    block_labels = []
    for block in range(num_blocks):

        # sample support of next block
        tower_idx = np.random.choice(len(towers))
        pos, quat = towers[tower_idx][-1]

        # position block on top of support
        new_pos = pos[:2] + (pos[2] + .0201,)

        # instantiate block
        label = env._add_block(new_pos, quat, mass=2, side=CUBE_SIDE)
        block_labels.append(label)

        # update new top of tower
        towers[tower_idx].append( (new_pos, quat) )

    # let blocks settle
    env.settle(1.)

    # initialize goal poses
    goal_poses = {}
    for label in block_labels:
        goal_poses[label] = env.get_block_pose(label)

    # swap some poses to create a non-trivial goal
    for _ in range(num_swaps):
        a, b = np.random.choice(block_labels, size=2)
        goal_poses[a], goal_poses[b] = goal_poses[b], goal_poses[a]

    return env, goal_poses

def evaluate(env, goal_poses):

    # let blocks settle
    env.settle(1.)

    # calculate how many blocks are at their goal positions
    num_correct = 0
    loc_errors, rot_errors = [], []
    for label, (goal_loc, goal_quat) in goal_poses.items():

        # get actual pose
        loc, quat = map(np.array, env.get_block_pose(label))

        # "correct" means within tolerance of goal location
        if (np.fabs(loc - goal_loc) < CUBE_SIDE / 2).all():
            num_correct += 1

        # location error: distance to goal location
        loc_errors.append(np.linalg.norm(loc - goal_loc))

        # rotation error: angle of rotation to align orientations
        rot_errors.append(2*np.arccos(min(1., np.fabs(quat @ goal_quat))))

    accuracy = num_correct / len(goal_poses)

    return accuracy, loc_errors, rot_errors

if __name__ == "__main__":

    # initialize controller class
    controller = Controller()

    # sample a validation trial
    env, goal_poses = sample_trial(num_blocks=8, num_swaps=4)

    # check camera image
    rgba, _, _ = env.get_camera_image()
    pt.imshow(rgba)
    pt.show()

    # run the controller on the trial
    controller.run(env, goal_poses)

    # evaluate success
    accuracy, loc_errors, rot_errors = evaluate(env, goal_poses)

    env.close()

    print(f"\n{int(100*accuracy)}% of blocks near correct goal positions")
    print(f"mean|max location error = {np.mean(loc_errors):.3f}|{np.max(loc_errors):.3f}")
    print(f"mean|max rotation error = {np.mean(rot_errors):.3f}|{np.max(rot_errors):.3f}")
