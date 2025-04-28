import os, time
import pybullet as pb
from pybullet_data import getDataPath
import itertools as it
import numpy as np

BASE_Z = 0.05715 # 2.25 inches

# helpers to convert between versor conventions
# pybullet is (x,y,z,w), ours is (w,x,y,z)
def _quat_to_pb(quat): return quat[1:] + (quat[0],)
def _pb_to_quat(quat): return (quat[3],) + quat[:3]

class SimulationEnvironment(object):

    def _load_urdf(self):
        fpath = os.path.dirname(os.path.abspath(__file__))
        pb.setAdditionalSearchPath(fpath)
        print(fpath)
        robot_id = pb.loadURDF(
            'poppy_ergo_jr.pybullet.urdf',
            basePosition = (0, 0, BASE_Z),
            baseOrientation = pb.getQuaternionFromEuler((0,0,0)),
            useFixedBase=True)
        return robot_id

    def __init__(self, show=True):
        """
        Initializes simulation environment with given parameters
        show: if True, visualize the simulation, otherwise run in background (faster)
        """

        self.show = show
        self.timestep = 1/240
        self.control_mode = pb.POSITION_CONTROL
        self.control_period = 1

        # setup world
        self.client_id = pb.connect(pb.GUI if show else pb.DIRECT)
        if show: pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
        pb.setTimeStep(self.timestep)
        pb.setGravity(0, 0, -9.81)
        pb.setAdditionalSearchPath(getDataPath())
        pb.loadURDF("plane.urdf")

        # setup robot        
        self.robot_id = self._load_urdf()
        self.num_joints = pb.getNumJoints(self.robot_id)
        self.joint_name, self.joint_index, self.joint_fixed = [], {}, []
        for i in range(self.num_joints):
            info = pb.getJointInfo(self.robot_id, i)
            name = info[1].decode('UTF-8')

            self.joint_name.append(name)
            self.joint_index[name] = i
            self.joint_fixed.append(info[2] == pb.JOINT_FIXED)

        # setup block data
        self.block_colors = list(it.product([0,1], repeat=3))
        self.block_id = {}

        # add a block for the arm platform (0 mass is static)
        self._add_block((0, .0682, -.1066 + BASE_Z), (0,0,0,1), mass = 0, side=0.2032)

        # save initial state for resets
        self.initial_state_id = pb.saveState(self.client_id)

        # reasonable initial viewpoint for arm
        pb.resetDebugVisualizerCamera(
            1.2000000476837158, 56.799964904785156, -22.20000648498535,
            (-0.6051651835441589, 0.26229506731033325, -0.24448847770690918))

    def _reset(self):
        # pb.resetSimulation()
        pb.restoreState(stateId = self.initial_state_id)
    
    def close(self):
        """
        Call when simulation is complete to avoid memory leaks
        """
        pb.disconnect()

    def get_joint_info(self):
        """
        Return joint_info, a tuple of joint information
        joint_info[i] = (joint_name, parent_index, translation, orientation, axis) for ith joint
        joint_name: string identifier for joint
        parent_index: index in joint_info of parent (-1 is base)
        translation: (3,) translation vector in parent joint's local frame
        orientation: (4,) orientation quaternion in parent joint's local frame
        axis: (3,) rotation axis vector in ith joint's own local frame (None for fixed joints)
        """
        return (
            ('m1', -1, (0.0, 0.0, 0.0327993216120967), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ('m2', 0, (0.0, 0.0, 0.0240006783879033), (0.7071067811865462, 0.0, -0.7071067811865488, 0.0), (0.0, 0.0, -1.0)),
            ('m3', 1, (0.054, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
            ('m4', 2, (0.045, 0.0, 0.0), (-0.7071067811865462, 0.0, 0.7071067811865488, 0.0), (0.0, 0.0, -1.0)),
            ('m5', 3, (0.0, -0.048, 0.0), (0.7071067811865462, 0.0, -0.7071067811865488, 0.0), (0.0, 0.0, 1.0)),
            ('t7f', 4, (0.0, -0.125, 0.0155), (1.0, 0.0, 0.0, 0.0), None), # fixed finger tip
            ('m6', 4, (0.0, -0.058, 0.0), (0.7071067811865462, 0.0, -0.7071067811865488, 0.0), (0.0, 0.0, -1.0)), # gripper motor
            ('t7m', 6, (-0.0155, -0.0675, 0.0), (1.0, 0.0, 0.0, 0.0), None), # movable finger tip
        )

    def _add_block(self, loc, quat, mass=2, side=.02):
        """
        Add a new block to the environment with given parameters
        loc: (3,) location vector in world frame
        quat: (4,) orientation quaternion in world frame
        mass: the mass of the block (0 is fixed in place)
        side: the length of each side (block is a cube)
        returns a string label for the block
        """

        # setup new label
        b = len(self.block_id)
        block = f"b{b:01d}"

        # setup pybullet multibody
        half_exts = (side*.5,)*3
        rgba = self.block_colors[b % len(self.block_colors)] + (1,)
        cube_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half_exts)
        cube_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=half_exts, rgbaColor=rgba)

        # add to environment
        self.block_id[block] = pb.createMultiBody(
            mass, cube_collision, cube_visual,
            basePosition=loc, baseOrientation=_quat_to_pb(quat))

        # return new block label
        return block

    def _get_base(self):
        loc, quat = pb.getBasePositionAndOrientation(self.robot_id)
        return (loc, _pb_to_quat(quat))

    def get_block_pose(self, label):
        """
        returns pose = (loc, quat) of block with given label
        loc: (x,y,z) coordinates of block
        quat: (w,x,y,z) where (x,y,z) is the axis of rotation and w is the "real" part
        """        
        loc, quat = pb.getBasePositionAndOrientation(self.block_id[label])
        return (loc, _pb_to_quat(quat))

    def _step(self, action):
        pb.setJointMotorControlArray(
            self.robot_id,
            jointIndices = range(len(self.joint_index)),
            controlMode = self.control_mode,
            targetPositions = action,
            targetVelocities = [0]*len(action),
            positionGains = [.25]*len(action), # important for position accuracy
        )
        for _ in range(self.control_period):
            pb.stepSimulation()

    # get/set joint angles as np.array
    def _get_position(self):
        states = pb.getJointStates(self.robot_id, range(len(self.joint_index)))
        return np.array([state[0] for state in states])    

    # convert a pypot style dictionary {... name:angle ...} to joint angle array
    # if convert == True, convert from degrees to radians
    def _angle_array_from(self, angle_dict, convert=True):
        angle_array = np.zeros(self.num_joints)
        for name, angle in angle_dict.items():
            angle_array[self.joint_index[name]] = angle
        if convert: angle_array *= np.pi / 180
        return angle_array

    # convert back to dict from array
    def _angle_dict_from(self, angle_array, convert=True):
        return {
            name: angle_array[j] * (180/np.pi if convert else 1)
            for j, name in enumerate(self.joint_name)
            if not self.joint_fixed[j]}

    def get_current_angles(self):
        """
        Get PyPot-style dictionary of current angles
        Returns angle_dict where angle_dict[joint_name] == angle (in degrees)
        """
        return self._angle_dict_from(self._get_position())

    def goto_position(self, target, duration=1.):
        """
        PyPot-style method that commands the arm to given target joint angles
        target is a dictionary where target[joint_name] == angle (in degrees)
        duration: time in seconds for the motion to complete
        """

        # get current/target angle arrays
        current = self._get_position()
        target = self._angle_array_from(target)

        # linearly interpolate trajectory
        num_steps = int(duration / (self.timestep * self.control_period) + 1)
        weights = np.linspace(0, 1, num_steps).reshape(-1,1)
        trajectory = weights * target + (1 - weights) * current

        # run trajectory
        for a, action in enumerate(trajectory): self._step(action)

    def settle(self, seconds=1.):
        """
        Runs simulation for given number of seconds to let blocks settle
        Arm is kept fixed at current position
        """
        action = self._get_position()
        num_steps = int(seconds / self.timestep)
        for _ in range(num_steps): self._step(action)
    
    def get_camera_image(self):
        """
        Returns current RGBA image array of shape (height, width, 4)
        also returns PyBullet camera view and projection matrices
        Note that PyBullet image capture is fairly time-consuming
        """
        width, height = 128, 96
        view = pb.computeViewMatrix(
            cameraEyePosition=(0, -.025, BASE_Z + .02),
            cameraTargetPosition=(0, -.4, 0), # focal point
            cameraUpVector=(0,0,.5),
        )
        proj = pb.computeProjectionMatrixFOV(
            fov=135,
            aspect=height/width,
            nearVal=.005,
            farVal=1.,
        )
        # rgba shape is (height, width, 4)
        _, _, rgba, _, _ = pb.getCameraImage(
            width, height, view, proj,
            flags = pb.ER_NO_SEGMENTATION_MASK) # not much speed difference
        rgba = np.array(rgba).reshape((height, width, 4))
        return rgba, view, proj

if __name__ == '__main__':

    import matplotlib.pyplot as pt

    env = SimulationEnvironment()

    joint_info = env.get_joint_info()
    current = env.get_current_angles()
    for info in joint_info: print(info)
    print(current)

    target = {motor: 20*np.random.randn() for motor in current.keys()}
    env.goto_position(target, duration=10.)

    rgba, _, _ = env.get_camera_image()
    print(rgba.shape)
    pt.imshow(rgba)
    pt.show()

    env.close()



