from simulation import SimulationEnvironment
import pybullet as pb
import numpy as np
import math
import time
import heapq

def get_active_joints(env):
    """
    Returns the indices of active (movable) joints based on joint axis info.
    """
    joint_info = env.get_joint_info()
    active = []
    for i, joint in enumerate(joint_info):
        if joint[4] is not None:  # joint axis exists → it's an active joint
            active.append(i)
    return active
    
def set_joint_positions_direct(env, joint_angles_rad, active_joints):
    for idx, joint_index in enumerate(active_joints):
        pb.resetJointState(env.robot_id, joint_index, joint_angles_rad[idx])

def check_robot_collision(env, joint_angles_rad, active_joints):
    try:
        set_joint_positions_direct(env, joint_angles_rad, active_joints)
        pb.performCollisionDetection()
        self_contacts = pb.getContactPoints(bodyA=env.robot_id, bodyB=env.robot_id)
        env_contacts = pb.getContactPoints(bodyA=env.robot_id)
        if self_contacts or env_contacts:
            print("Collision detected!")
            if self_contacts:
                print("Self-collision between links")
                for contact in self_contacts:
                    link_a, link_b = contact[3], contact[4]
                    name_a = pb.getJointInfo(env.robot_id, link_a)[1].decode() if link_a < env.num_joints else f"Link {link_a}"
                    name_b = pb.getJointInfo(env.robot_id, link_b)[1].decode() if link_b < env.num_joints else f"Link {link_b}"
                    print(f"Links {name_a} (index {link_a}) and {name_b} (index {link_b}) collided")
            if env_contacts:
                for contact in env_contacts:
                    body_b, link_a = contact[2], contact[3]
                    name_a = pb.getJointInfo(env.robot_id, link_a)[1].decode() if link_a < env.num_joints else f"Link {link_a}"
                    pos = pb.getLinkState(env.robot_id, link_a)[0]
                    print(f"Robot link {name_a} (index {link_a}) collided with body ID: {body_b}")
                    print(f"Link {name_a} position: {pos}")
        return len(self_contacts) > 0 or len(env_contacts) > 0
    except pb.error as e:
        print(f"PyBullet error in check_robot_collision: {e}")
        return True  # Assume collision if server is disconnected

def semiRandomSample(steer_goal_p, q_goal, num_dof, joint_limits):
    q_goal = q_goal[:num_dof]
    uniform_sample = np.array([
        np.random.uniform(joint_limits[i][0], joint_limits[i][1])
        for i in range(num_dof)
    ])
    output = np.random.choice([0, 1], p=[1 - steer_goal_p, steer_goal_p])
    return q_goal if output == 0 else uniform_sample

def L2(v1, v2): #euclidean distance
    res = 0
    for i in range(6):
        res += (abs(v1[i] - v2[i]))**2
    return math.sqrt(res)

def nearest(vertices, q_rand):
    min_dist = L2(vertices[0], q_rand)
    nearest_vertex = vertices[0]
    for v in vertices:
        curr_dist = L2(v, q_rand)
        if curr_dist < min_dist:
            min_dist = curr_dist
            nearest_vertex = v
    return nearest_vertex

def steer(q_nearest, q_rand, delta_q):
    distance = L2(q_rand, q_nearest)
    if distance == 0:
        return q_nearest
    coefficient = delta_q / distance
    q_new = q_nearest + (q_rand - q_nearest) * min(coefficient, 1.0)
    return q_new

def obstacleFree(env, q_nearest, q_new, active_joints):
    return not check_robot_collision(env, q_new, active_joints)

def createGraph(vertices, edges):
    graph = {}
    vertex_tuples = [tuple(v) for v in vertices]
    for v in vertex_tuples:
        graph[v] = []
    for parent, children in edges.items():
        parent_tup = tuple(parent)
        for child in children:
            child_tup = tuple(child)
            if parent_tup in graph and child_tup in graph:
                weight = L2(np.array(parent_tup), np.array(child_tup))
                graph[parent_tup].append((child_tup, weight))
                graph[child_tup].append((parent_tup, weight))  # Undirected
    print("Graph vertices:", list(graph.keys())[:5])
    return graph

def dijkstra_shortest_path(graph, start, end):
    start_tup = tuple(start)
    end_tup = tuple(end)
    queue = [(0, start_tup, [start_tup])]
    visited = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)
            if node == end_tup:
                return [np.array(p) for p in path]
            for neighbor, weight in graph[node]:
                if neighbor not in visited:
                    new_cost = cost + weight
                    new_path = path + [neighbor]
                    print(f"Pushing to queue: cost={new_cost}, neighbor={neighbor}")
                    heapq.heappush(queue, (new_cost, neighbor, new_path))
    return None

def cost_to_root(node, parent_map):
    cost = 0
    node_tup = tuple(node)
    while node_tup in parent_map:
        parent = parent_map[node_tup]
        cost += L2(node, parent)
        node = parent
        node_tup = tuple(node)
    return cost

def get_neighbors(vertices, q_new, radius):
    return [v for v in vertices if L2(v, q_new) < radius]

def calculatePath(vertices, edges, start, end):
    graph = createGraph(vertices, edges)
    return dijkstra_shortest_path(graph, start, end)

def interpolate_configs(q1, q2, steps=100):
    return [q1 + (q2 - q1) * t / steps for t in range(steps + 1)]

def smooth_path(path, env, active_joints, max_attempts=2000):
    if not path or len(path) < 3:
        return path
    smoothed = path.copy()
    for _ in range(max_attempts):
        i, j = sorted(np.random.choice(len(smoothed), 2, replace=False))
        if j - i <= 1:
            continue
        if obstacleFree(env, smoothed[i], smoothed[j], active_joints):
            smoothed = smoothed[:i+1] + smoothed[j:]
    final_path = []
    for i in range(len(smoothed) - 1):
        interp_configs = interpolate_configs(smoothed[i], smoothed[i+1], steps=100)
        for config in interp_configs:
            if not obstacleFree(env, smoothed[i], config, active_joints):
                break
            final_path.append(config)
    # Prune redundant configurations
    pruned_path = [final_path[0]]
    for config in final_path[1:]:
        if L2(pruned_path[-1], config) > 0.005:
            pruned_path.append(config)
    print("Final smoothed path endpoints:", pruned_path[0], pruned_path[-1])
    return pruned_path

def rrt_star(q_init, q_goal, env, MAX_ITERS=20000, delta_q=0.001, steer_goal_p=0.01, num_dof=6, active_joints=None):
    vertices = [q_init]
    parent_map = {}
    edges = {}
    goal_threshold = 0.05
    neighbor_radius = 0.2 * np.sqrt(num_dof)

    for _ in range(MAX_ITERS):
        q_rand = semiRandomSample(steer_goal_p, q_goal, num_dof, env.joint_limits)
        q_nearest = nearest(vertices, q_rand)
        q_new = steer(q_nearest, q_rand, delta_q)

        if not obstacleFree(env, q_nearest, q_new, active_joints):
            continue

        neighbors = get_neighbors(vertices, q_new, neighbor_radius)
        min_cost = cost_to_root(q_nearest, parent_map) + L2(q_new, q_nearest)
        best_parent = q_nearest

        for neighbor in neighbors:
            if obstacleFree(env, neighbor, q_new, active_joints):
                cost = cost_to_root(neighbor, parent_map) + L2(q_new, neighbor)
                if cost < min_cost:
                    min_cost = cost
                    best_parent = neighbor

        parent_map[tuple(q_new)] = best_parent
        vertices.append(q_new)
        edges.setdefault(tuple(best_parent), []).append(q_new)
        edges.setdefault(tuple(q_new), []).append(best_parent)

        for neighbor in neighbors:
            if obstacleFree(env, q_new, neighbor, active_joints):
                old_cost = cost_to_root(neighbor, parent_map)
                new_cost = min_cost + L2(q_new, neighbor)
                if new_cost < old_cost:
                    parent_map[tuple(neighbor)] = q_new
                    edges.setdefault(tuple(q_new), []).append(neighbor)
                    edges.setdefault(tuple(neighbor), []).append(q_new)

        if L2(q_new, q_goal) < goal_threshold and obstacleFree(env, q_new, q_goal, active_joints):
            parent_map[tuple(q_goal)] = q_new
            vertices.append(q_goal)
            edges.setdefault(tuple(q_new), []).append(q_goal)
            edges.setdefault(tuple(q_goal), []).append(q_new)

            path = calculatePath(vertices, edges, q_init, q_goal)
            return path

    return None

def execute_path(path_conf, env, visualize=False, drop=False, interp_steps=200):
    if not path_conf:
        return
    interpolated_path = []
    for i in range(len(path_conf) - 1):
        interp_configs = interpolate_configs(path_conf[i], path_conf[i+1], interp_steps)
        for config in interp_configs:
            if not obstacleFree(env, path_conf[i], config, active_joints):
                break
            interpolated_path.append(config)
    print("Interpolated path length:", len(interpolated_path))
    if visualize:
        trail = []
    print("Joint index mapping:", env.joint_index)
    for i, joint_angles in enumerate(interpolated_path):
        if i % 100 == 0:
            print(f"Processing config {i}/{len(interpolated_path)}")
        full_angles = np.zeros(env.num_joints)
        for idx, joint_index in enumerate(active_joints):
            full_angles[joint_index] = joint_angles[idx]
        print(f"Full angles (radians): {full_angles}")
        # Manually compute expected joint dict
        expected_dict = {f'm{idx+1}': np.degrees(joint_angles[idx]) for idx in range(len(joint_angles)-1)}
        expected_dict['m6'] = np.degrees(joint_angles[-1])  # m6 is index 6
        joint_dict = env._angle_dict_from(np.degrees(full_angles))
        print(f"Config {i}: {joint_angles} -> Expected dict: {expected_dict}, Joint dict: {joint_dict}")
        env.goto_position(joint_dict, duration=0.005)
        applied_angles = env._get_position()
        applied_dict = env._angle_dict_from(applied_angles)
        print(f"Applied Config {i}: {applied_dict}")
        # Check for angle deviations
        for key, val in expected_dict.items():
            if abs(val - applied_dict.get(key, 0)) > 0.01:
                print(f"Warning: Joint {key} deviation: Expected {val}, Applied {applied_dict.get(key, 0)}")
        if visualize and i % 10 == 0:
            pos = pb.getLinkState(env.robot_id, 6)[0]  # m6 link
            pb.addUserDebugLine(pos, pos, [0, 1, 0], lineWidth=5, lifeTime=0.5)
        pb.stepSimulation()
    if drop:
        current = env.get_current_angles()
        current['m6'] = 15
        env.goto_position(current, duration=1.0)
        current['m6'] = -15
        env.goto_position(current, duration=1.0)

if __name__ == "__main__":
    env = SimulationEnvironment(show=True)
    active_joints = get_active_joints(env)
    num_dof = len(active_joints)

    print("Active joints:", active_joints)
    for idx in active_joints:
        joint_name = pb.getJointInfo(env.robot_id, idx)[1].decode()
        lower, upper = pb.getJointInfo(env.robot_id, idx)[8:10]
        print(f"Joint {idx}: {joint_name}, Limits: [{lower}, {upper}]")

    joint_limits = []
    for idx in active_joints:
        if idx == 1:  # m2
            lower, upper = -np.pi/3, np.pi/3
        else:
            lower, upper = -np.pi/2, np.pi/2
        joint_limits.append((lower, upper))
    env.joint_limits = joint_limits
    print("Corrected joint limits:", joint_limits)

    full_start = env._get_position()
    start = full_start[active_joints]
    start_dict = env._angle_dict_from(np.degrees(full_start))
    print("Start configuration (degrees):", start_dict)

    goal_dict = {'m1': 0, 'm2': -45, 'm3': -30, 'm4': 0, 'm5': 0, 'm6': 0}
    print("Checking goal against limits...")
    if goal_dict['m2'] * np.pi/180 > np.pi/3 or goal_dict['m2'] * np.pi/180 < -np.pi/3:
        print(f"Goal m2 ({goal_dict['m2']}°) exceeds limit [-π/3, π/3]. Clamping...")
        goal_dict['m2'] = min(max(goal_dict['m2'], -60), 60)
    goal_full = env._angle_array_from(goal_dict)
    print("Goal_full (radians):", goal_full)
    goal = goal_full[active_joints]
    goal_dict_rad = env._angle_dict_from(goal_full)
    print("Goal configuration (degrees):", goal_dict_rad)

    print("Start (radians):", start)
    print("Goal (radians):", goal)
    print("L2 distance:", L2(start, goal))

    if L2(start, goal) < 0.05:
        print("Start and goal are too close. No motion required.")
        env.close()
        exit()

    path = rrt_star(
        start, goal, env, MAX_ITERS=20000, delta_q=0.001, steer_goal_p=0.01,
        num_dof=num_dof, active_joints=active_joints
    )

    if path:
        print("Path found. Smoothing...")
        smoothed_path = smooth_path(path, env, active_joints)
        print("Path length:", len(smoothed_path))
        for i, config in enumerate(smoothed_path):
            print(f"Smoothed Config {i}: {config}")
        for i in range(len(smoothed_path) - 1):
            diff = L2(smoothed_path[i], smoothed_path[i+1])
            print(f"Step {i} to {i+1}: L2 distance = {diff}")
        print("Executing smoothed path...")
        execute_path(smoothed_path, env, visualize=True, drop=True)
    else:
        print("No path found.")

    env.close()