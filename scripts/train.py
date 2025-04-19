import rclpy
import rclpy.logging, rclpy.time
from rclpy.node import Node
import numpy as np
import threading, itertools, math, torch, argparse, datetime, os, multiprocessing
from TD_CBAM2 import TD3
from lidar_preprocessing import LidarPreprocessing
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from geometry_msgs.msg import Twist
from sensor_msgs_py import point_cloud2 as pc2
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float32MultiArray
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni
from omni.isaac.core import World, SimulationContext
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils import stage as stage_utils
from omni.isaac.core.objects import DynamicCylinder
from pxr import UsdGeom, UsdPhysics, Gf, PhysxSchema, Sdf

scripts_path = os.path.abspath(os.path.dirname(__file__))
pkg_path = os.path.dirname(scripts_path)
usd_file_path = os.path.join(pkg_path, "usd/train_model.usd")

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)
world = World(stage_units_in_meters=1.0, physics_dt=1/500, rendering_dt=1/60)
omni.usd.get_context().open_stage(usd_file_path)

simulation_app.update()
while stage_utils.is_stage_loading():
    simulation_app.update()

simulation_context = SimulationContext()
simulation_context.initialize_physics()
simulation_context.play()

class IsaacEnvROS2(Node):
    def __init__(self, size):
        super().__init__('env')
        self.size = size
        self.offset = 1.0
        
        # Basic Parameters Settings
        self.cmd_vel = Twist()
        self.pose = np.zeros(3, dtype=np.float32)
        self.start = np.zeros(2, dtype=np.float32)
        self.goal = np.array([self.size, self.size], dtype=np.float32)
        self.euclidean_dist = np.linalg.norm(self.goal - self.start)
        self.max_vel = 1.0
        self.min_vel = 0.5
        self.max_omega = 0.7853981633974483
        self.COLLISION_THRESHOLD = 0.4
        self.collision_bool = False
        self.done = False
        self.cylinder_coords = []
        self.remain_dist = self.euclidean_dist
        
        # TD3 Network Settings
        self.voxel_data = torch.from_numpy(np.zeros((180, 10, 3), dtype=np.float32))
        self.goal_state = torch.from_numpy(np.array([self.goal[0] - self.pose[0], self.goal[1] - self.pose[1]]))
        self.next_state = (self.voxel_data, self.goal_state)
        self.next_state_add = self.goal - self.pose[:2]
        
        # ROS2 Basic Settings
        self.qos = QoSProfile(depth=10, reliability = ReliabilityPolicy.RELIABLE, durability = DurabilityPolicy.VOLATILE)
        self.lidar_subscription = self.create_subscription(Float32MultiArray, '/voxel_data', self.lidar_cb, self.qos)
        self.local_min_dist_subscription = self.create_subscription(Float32MultiArray, '/local_min_dist', self.min_dist_cb, self.qos)
        self.publisher_cmd_vel = self.create_publisher(Twist, '/cmd_vel', self.qos)
        self.publisher_cylinder_coords = self.create_publisher(Float32MultiArray, '/cylinder_coords', self.qos)
        self.publisher_robot_pose = self.create_publisher(Float32MultiArray, '/robot_pose', self.qos)
        
        self.create_timer(0.01, self.cmd_cb)
        self.create_timer(0.01, self.robot_pose)
        self.create_timer(0.01, self.collision_detect)

    def robot_pose(self):
        try:
            robot_prim_path = "/World/mobile_manipulator/base_link/base_body"
            robot_prim = world.stage.GetPrimAtPath(robot_prim_path)
            if robot_prim.IsValid():
                xform = UsdGeom.Xformable(robot_prim)
                transform_matrix = xform.GetLocalTransformation()
                translation = transform_matrix.ExtractTranslation()
                rotation = transform_matrix.ExtractRotation()
                _, _, yaw = self.axis_angle_to_euler(rotation)
                
                self.pose = np.array([translation[0], translation[1], yaw], dtype=np.float32)
                self.prev_remain_dist = self.remain_dist
                self.remain_dist = np.linalg.norm(self.goal - self.pose[:2])
                self.publisher_robot_pose.publish(Float32MultiArray(data=[self.pose[0], self.pose[1], yaw]))
            else:
                print(f"Error: Robot prim not found at {robot_prim_path}")
        except Exception as e:
            print(f"Failed to update robot pose from sim: {str(e)}")

    def axis_angle_to_euler(self, rotation):
        axis = rotation.axis
        angle = rotation.angle
        
        w = np.cos(angle/2)
        x = axis[0] * np.sin(angle/2)
        y = axis[1] * np.sin(angle/2)
        z = axis[2] * np.sin(angle/2)
        
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        
        # Normalize angles to be within -pi to pi
        roll = math.atan2(math.sin(roll), math.cos(roll))
        pitch = math.atan2(math.sin(pitch), math.cos(pitch))
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        
        return roll, pitch, yaw

    def cmd_cb(self):
        self.publisher_cmd_vel.publish(self.cmd_vel)

    def min_dist_cb(self, msg):
        self.local_min_dist = msg.data[0]

    def lidar_cb(self, msg):
        self.angle_bins = msg.layout.dim[0].size
        self.z_bins = msg.layout.dim[1].size
        self.angle_min = -math.pi/2
        self.angle_max = math.pi/2
        self.angle_bin_size = (self.angle_max - self.angle_min) / self.angle_bins
        self.z_min = -0.2
        self.z_max = 2.8
        self.z_bin_size = (self.z_max - self.z_min) / self.z_bins
        self.voxel_data = torch.from_numpy(np.array(msg.data, dtype=np.float32).reshape(self.angle_bins, self.z_bins, 3))
        self.next_state = self.voxel_data

    def collision_detect(self):
        if self.cylinder_coords:
            for coords in self.cylinder_coords:
                dist = np.linalg.norm(np.array(coords[:2]) - self.pose[:2])
                if dist < self.COLLISION_THRESHOLD:
                    self.cmd_vel.linear.x = 0.0
                    self.cmd_vel.angular.z = 0.0
                    self.local_min_dist = dist
        
        if self.pose is not None and self.pose.size > 0:
            if not ((-0.6 <= self.pose[0] <= 5.6) and (-0.6 <= self.pose[1] <= 5.6)):
                self.cmd_vel.linear.x = 0.0
                self.cmd_vel.angular.z = 0.0
    
    def parameters_reset(self):
        self.done = False
        self.collision_bool = False
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.angular.z = 0.0
        self.cylinder_coords = []
    
    def reset(self):
        simulation_context.reset()
        self.parameters_reset()
        wall_height, wall_thickness, bias  = 3.0, 1.0, 2
        wall_nums = 0
        for wall_name in ["bottom", "top", "left", "right"]:
            wall_prim_path = f"/World/Wall/{wall_name}_wall"
            wall_xform = XFormPrim(wall_prim_path)
            cube_geom = UsdGeom.Cube.Define(world.stage, f"{wall_prim_path}/Cube_{wall_nums+1}")
            cube_geom.GetSizeAttr().Set(1.0)
            UsdPhysics.RigidBodyAPI.Apply(world.stage.GetPrimAtPath(wall_prim_path))
            UsdPhysics.CollisionAPI.Apply(world.stage.GetPrimAtPath(wall_prim_path))

            if wall_name == "bottom":
                position = [ -bias + wall_thickness / 2, self.size / 2, wall_height / 2]
                scale = [wall_thickness, self.size + 2 * bias, wall_height]
            elif wall_name == "top":
                position = [self.size + bias - wall_thickness / 2, self.size / 2, wall_height / 2]
                scale = [wall_thickness, self.size + 2 * bias, wall_height]
            elif wall_name == "left":
                position = [self.size / 2, self.size + bias - wall_thickness / 2, wall_height / 2]
                scale = [self.size + bias, wall_thickness, wall_height]
            elif wall_name == "right":
                position = [self.size / 2, -bias + wall_thickness / 2, wall_height / 2]
                scale = [self.size + bias, wall_thickness, wall_height]
            
            wall_xform.set_world_pose(position)
            wall_xform.set_local_scale(scale)
            wall_nums += 1
        
        cylinder_xforms = []
        one_side_parts = 4
        region_size = self.size / one_side_parts
        cylinder_radius = self.size / 50
        cylinder_height = 2.0
        obstacle_num = 4
        for i in range(obstacle_num):
            cylinder_prim_path = f"/World/Obstacles/Cylinder_{i}"
            try:
                cylinder_xform = XFormPrim(cylinder_prim_path)
                cylinder_xforms.append(cylinder_xform)
                cylinder_geom = UsdGeom.Cylinder.Define(world.stage, f"{cylinder_prim_path}/Cylinder_{i}")
                cylinder_geom.GetRadiusAttr().Set(cylinder_radius)
                cylinder_geom.GetHeightAttr().Set(cylinder_height)
                UsdPhysics.RigidBodyAPI.Apply(world.stage.GetPrimAtPath(cylinder_prim_path))
                UsdPhysics.CollisionAPI.Apply(world.stage.GetPrimAtPath(cylinder_prim_path))
                
                coords = [
                    i*region_size + np.random.uniform(0, region_size),
                    np.random.uniform(region_size, self.size - region_size),
                    cylinder_height / 2
                ]
                cylinder_xform.set_world_pose(coords, [0, 0, 0, 1])
                self.cylinder_coords.append(coords)
                self.next_state_add = self.goal - self.pose[:2]
            except Exception as e:
                self.get_logger().error(f"Error creating/resetting Cylinder_{i}:  {str(e)}")
        self.publish_cylinder_coords()
        return self.next_state, self.next_state_add
    
    def publish_cylinder_coords(self):
        try:
            msg = Float32MultiArray()
            data = []
            for coords in self.cylinder_coords:
                data.extend(coords[:2])
            msg.data = data
            self.publisher_cylinder_coords.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing cylinder coordinates: {str(e)}")
    
    def step(self, action, time_steps, max_episode_steps):
        self.done = False
        
        self.omega = action[0]
        if np.abs(action[0]) > self.max_omega:
            self.omega = np.sign(action[0]) * self.max_omega
        
        self.vel = self.max_vel - (abs(self.omega) / self.max_omega) * (self.max_vel - self.min_vel)
        self.cmd_vel.linear.x = self.vel
        self.cmd_vel.angular.z = float(self.omega)
        
        for _ in range(20):
            simulation_context.step(render=True)
        
        self.next_state_add = self.goal - self.pose[:2]
        
        if self.local_min_dist > self.COLLISION_THRESHOLD:
            self.collision_bool = False
        else:
            self.collision_bool = True
        reward, self.done = self.reward_function_test_one(time_steps, max_episode_steps)
        return self.next_state, self.next_state_add, reward, self.done
            
    def reward_function(self, time_steps, max_episode_steps):
        time_threshold = 50
        sigma = self.euclidean_dist / 3
        d_kappa = 100
        t_kappa = d_kappa / 10
        additional_reward = d_kappa
        beta = self.euclidean_dist / time_threshold
        delta = self.euclidean_dist - self.remain_dist - beta * time_steps
        
        dist_efficiency = d_kappa * np.exp(-self.remain_dist**2 / (2 * sigma**2))
        time_efficiency = t_kappa * np.sign(delta) * delta**2
        if delta < 0:
            time_efficiency *= 2
        
        reward = dist_efficiency + time_efficiency
        if reward <= -100 or time_steps >= max_episode_steps:
            self.get_logger().info(f"TIME OVER! DONE.")
            reward -= additional_reward
            self.done = True
        if self.collision_bool == True:
            self.get_logger().info(f"COLLISION OCCUR! DONE.")
            # reward -= additional_reward * (1 + min(time_threshold - time_steps, 0) / time_threshold)
            reward -= additional_reward * (1 + ((time_threshold/2) / (time_steps + 1)))
            self.done = True
        elif self.remain_dist < np.sqrt(2):
            self.get_logger().info(f"GOAL!! DONE.")
            reward += 5 * additional_reward
            self.done = True
        self.get_logger().info(f"Reward: {reward:.2f}, Dist Reward : {dist_efficiency:.2f}, Time Reward : {time_efficiency:.2f}")
        return reward, self.done
    
    def reward_function_test_one(self, time_steps, max_episode_steps):
        time_threshold = 50
        sigma = self.euclidean_dist / 3
        kappa_d = 100
        kappa_t = kappa_d / 10
        kappa_p = kappa_d
        additional_reward = kappa_d
        beta = self.euclidean_dist / time_threshold
        
        dist_efficiency = kappa_d * np.exp(-self.remain_dist**2 / (2 * sigma**2))
        
        time_delta = self.euclidean_dist - self.remain_dist - beta * time_steps
        time_efficiency = kappa_t * np.sign(time_delta) * time_delta**2
        if time_delta < 0:
            time_efficiency *= 2
        
        progress_delta = self.remain_dist - self.prev_remain_dist
        progress_efficiency = kappa_p * progress_delta
        # if progress_delta < 0:
        #     progress_efficiency *= 2
        
        reward = dist_efficiency + time_efficiency + progress_efficiency
        if reward <= -100 or time_steps >= max_episode_steps:
            self.get_logger().info(f"TIME OVER! DONE.")
            reward -= additional_reward
            self.done = True
        if self.collision_bool == True:
            self.get_logger().info(f"COLLISION OCCUR! DONE.")
            reward -= additional_reward * (1 + min(time_threshold - time_steps, 0) / time_threshold)
            self.done = True
        elif self.remain_dist < np.sqrt(2):
            self.get_logger().info(f"GOAL!! DONE.")
            reward += 5 * additional_reward
            self.done = True
        print(f"Reward: {reward:.2f}, Dist Reward : {dist_efficiency:.2f}, Time Reward : {time_efficiency:.2f}, Progress Reward : {progress_efficiency:.2f}")
        return reward, self.done
    
    def reward_function_test_two(self, time_steps, max_episode_steps):
        time_threshold = 50
        sigma = self.euclidean_dist / 3
        d_kappa = 100
        additional_reward = d_kappa
        
        current_angle = self.pose[2]
        target_angle = np.arctan2(self.goal[1] - self.pose[1], self.goal[0] - self.pose[0])
        angle_diff = target_angle - current_angle
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
        angle_diff = abs(angle_diff)
        
        angle_margin = np.pi / 4
        alignment_weight = d_kappa / 5
        
        alignment_penalty = alignment_weight * max(angle_diff - angle_margin, 0)
        dist_efficiency = d_kappa * np.exp(-self.remain_dist**2 / (2 * sigma**2))
        
        beta = self.euclidean_dist / time_threshold
        delta = self.euclidean_dist - self.remain_dist - beta * time_steps
        time_efficiency = (d_kappa / 10) * np.sign(delta) * delta**2
        if delta < 0:
            time_efficiency *= 2
        reward = dist_efficiency + time_efficiency - alignment_penalty
        
        if reward <= -100 or time_steps >= max_episode_steps:
            self.get_logger().info(f"TIME OVER! DONE.")
            reward -= additional_reward
            self.done = True
        if self.collision_bool == True:
            self.get_logger().info(f"COLLISION OCCUR! DONE.")
            reward -= additional_reward * (1 + ((time_threshold/2) / (time_steps + 1)))
            self.done = True
        elif self.remain_dist < np.sqrt(2):
            self.get_logger().info(f"GOAL!! DONE.")
            reward += 5 * additional_reward
            self.done = True
        print(f"Reward: {reward:.2f}, Dist: {dist_efficiency:.2f}, Time: {time_efficiency:.2f}, Alignment: {-alignment_penalty:.2f}")
        return reward, self.done

if __name__ == '__main__':
    rclpy.init(args=None)

    parser = argparse.ArgumentParser(description='TD3 Args')
    parser.add_argument('--env-name', default="obstacle_avoidance", help='quadruped_isaac')
    parser.add_argument('--policy', default="Gaussian", help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True, help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G', help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G', help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G', help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--policy_freq', type=bool, default=2, metavar='G', help='policy frequency for TD3 updates')
    parser.add_argument('--seed', type=int, default=123456, metavar='N', help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=12001, metavar='N', help='maximum number of steps (default: 5000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N', help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N', help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N', help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=500000, metavar='N', help='size of replay buffer (default: 10000000)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G', help='Automaically adjust α (default: False)')
    parser.add_argument('--cuda', action="store_true", default=True, help='run on CUDA (default: False)')
    args = parser.parse_args()

    size = 5.0
    resolution = 0.1
    
    env = IsaacEnvROS2(size)
    # path_planner = AStarPlanner(size, resolution)

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(env)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = env.create_rate(2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    action_space = np.ones(1)
    env.get_logger().info(f"state_space:{len(env.next_state)}")
    env.get_logger().info(f"action_space:{action_space}")
    env.get_logger().info(f"args : {args}")
    agent = TD3(len(env.next_state), action_space, args)
    file_name = "checkpoints"
    try:
        agent.load_checkpoint(pkg_path, file_name)
        env.get_logger().info("Checkpoint loaded successfully.")
    except:
        env.get_logger().info("No checkpoint found, start training from scratch.")

    #Tensorboard
    writer = SummaryWriter('runs/{}_TD3_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name, args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)
    memory.clear()
    # Training Loop
    total_numsteps = 0
    # updates = 0
    max_episode_steps = 100000
    
    try:
        while rclpy.ok():
            for i_episode in itertools.count(1):
                episode_reward = 0
                episode_steps = 0
                done = False
                
                state, state_add = env.reset()
                
                while not done:
                    print(f"episode step:{episode_steps}, in total:{total_numsteps}/{args.num_steps}")
                    if args.start_steps > total_numsteps:
                        action = np.random.uniform(-0.7853981633974483, 0.7853981633974483, 1)
                        # action = path_planner.plan()
                        # print(action)
                    else:
                        action = agent.select_action(state, state_add)
                        action += np.random.normal(0, 0.2, 1)

                    if len(memory) > args.batch_size:
                        av_critic_loss, av_Q, max_Q = agent.update_parameters(memory, args)

                        writer.add_scalar('av_critic_loss', av_critic_loss, total_numsteps)
                        writer.add_scalar('av_Q', av_Q, total_numsteps)
                        writer.add_scalar('max_Q', max_Q, total_numsteps)
                        
                    next_state, next_state_add, reward, done = env.step(action, episode_steps, max_episode_steps)
                    episode_steps += 1
                    total_numsteps += 1
                    episode_reward += reward

                    mask = 1 if episode_steps == max_episode_steps else float(not done)
                    memory.push(state, state_add, action, reward, next_state, next_state_add, mask)

                    state = next_state
                    state_add = next_state_add

                if total_numsteps > args.num_steps:
                    env.get_logger().info(f"Training finished after {total_numsteps} steps")
                    break

                writer.add_scalar('reward/train', episode_reward, i_episode)
                print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

                if i_episode % 20 == 0 and args.eval is True:
                    avg_reward = 0.
                    episodes = 20
                    for i  in range(episodes):
                        print(f"eval episode{i}")
                        state, state_add = env.reset()
                        episode_reward = 0
                        eval_steps = 0
                        done = False
                        while not done:
                            action = agent.select_action(state, state_add)
                            next_state, next_state_add, reward, done = env.step(action, eval_steps, max_episode_steps)
                            episode_reward += reward

                            eval_steps += 1
                            state = next_state
                        avg_reward += episode_reward
                    avg_reward /= episodes

                    writer.add_scalar('avg_reward/test', avg_reward, i_episode)

                    print("--------------------------------------------------------------------------------")
                    print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
                    print("--------------------------------------------------------------------------------")
                    
                if i_episode % 40 == 0:
                    agent.save_checkpoint(pkg_path, file_name, i_episode)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")