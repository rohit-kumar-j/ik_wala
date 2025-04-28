"""
Example controller to illustrate API methods and conventions
"""
import numpy as np
import matplotlib.pyplot as pt
from simulation import SimulationEnvironment
import evaluation as ev

class ExampleController:
    def __init__(self):
        # no optimized model data in this example
        pass

    def run(self, env, goal_poses):
        # run the controller in the environment to achieve the goal
        # this example ignores goal_poses
        # just follows a hand-coded trajectory for sake of example

        # get starting angles for trajectory
        init_angles = env.get_current_angles()
        trajectory = []

        # move arm down around block
        stage_angles = dict(init_angles)
        stage_angles["m2"] = 20. # second motor angle (degrees)
        stage_angles["m3"] = 35. # second motor angle (degrees)
        trajectory.append(stage_angles)

        # close the gripper (sixth motor)
        close_angles = dict(stage_angles)
        close_angles["m6"] = -20
        trajectory.append(close_angles)

        # lift arm
        lift_angles = dict(close_angles)
        lift_angles["m3"] = 20
        trajectory.append(lift_angles)

        # rotate arm
        rotate_angles = dict(lift_angles)
        rotate_angles["m1"] = -33
        trajectory.append(rotate_angles)

        # lower arm
        lower_angles = dict(rotate_angles)
        lower_angles["m3"] = 35
        trajectory.append(lower_angles)

        # release and lift arm
        release_angles = dict(lower_angles)
        release_angles["m3"] = 20
        release_angles["m6"] = 0
        trajectory.append(release_angles)

        # runs the trajectory
        duration = 5. # duration of each waypoint transition (in seconds)
        for waypoint in trajectory:
            env.goto_position(waypoint, duration)

if __name__ == "__main__":

    # initialize controller class
    controller = ExampleController()

    # launch environment, show=True visualizes the simulation
    # show=False is substantially faster for when you are training/testing
    env = SimulationEnvironment(show=True)

    # joint info is available for you to program forward/inverse kinematics
    joint_info = env.get_joint_info()
    for info in joint_info: print(info)
    input("[Enter] to continue ...")    

    # get the tower base positions used for evaluation
    # there are two concentric rings of towers
    # each ring has half_spots many towers
    # this example uses 2*3 towers, but real evaluation will have more
    tower_poses = ev.get_tower_base_poses(half_spots = 3)

    # add a block right in front of the robot
    # _add_block is a private method, you can you can use it here for testing
    # but your controller should not add blocks during evaluation
    loc, quat = tower_poses[1]
    loc = loc[:2] + (.011,) # increase z coordinate above floor with slight gap
    label = env._add_block(loc, quat, side=.02) # .02 is cube side length
    env.settle(1.) # run simulation for 1 second to let the block settle on the floor

    # you can get a synthetic camera image if you want to use it (not required)
    rgba, _, _ = env.get_camera_image()
    pt.imshow(rgba)
    pt.show()

    # a validation trial will have a dict of goal poses, one for each block
    # setup a goal one spot to the left
    loc, quat = tower_poses[0]
    loc = loc[:2] + (.01,) # increase z coordinate above floor
    goal_poses = {label: (loc, quat)}

    # run the controller on the trial
    controller.run(env, goal_poses)

    # evaluation metrics
    accuracy, loc_errors, rot_errors = ev.evaluate(env, goal_poses)

    # close any environment you instantiate to avoid memory leaks
    env.close()

    # display the metrics
    print(f"\n{int(100*accuracy)}% of blocks near correct goal positions")
    print(f"mean|max location error = {np.mean(loc_errors):.3f}|{np.max(loc_errors):.3f}")
    print(f"mean|max rotation error = {np.mean(rot_errors):.3f}|{np.max(rot_errors):.3f}")

    input("[Enter] to continue ...")

    # real evaluation should use randomly sampled trials
    # hand-coded trajectory will not work on these

    # sample a validation trial
    env, goal_poses = ev.sample_trial(num_blocks=5, num_swaps=1, show=True)

    # run the controller on the trial
    # copies goal_poses in case your controller modifies it (but you shouldn't)
    controller.run(env, dict(goal_poses))

    # evaluate success
    accuracy, loc_errors, rot_errors = ev.evaluate(env, goal_poses)

    env.close()

    print(f"\n{int(100*accuracy)}% of blocks near correct goal positions")
    print(f"mean|max location error = {np.mean(loc_errors):.3f}|{np.max(loc_errors):.3f}")
    print(f"mean|max rotation error = {np.mean(rot_errors):.3f}|{np.max(rot_errors):.3f}")

