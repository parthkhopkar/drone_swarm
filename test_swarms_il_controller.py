import time
import argparse
import random
import gym
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType

from swarms import Environment2D, Boid, Goal, Sphere
import swarms
import swarmnet

log = []
# The Z position at which to intitialize the drone
# When using velocity based PID control, same value of Z
# needs to be set in Base{Single,Multi}agentAviary.py
Z = 0.5
N = 5
# Number of entries in the list determines number of drones
# initial_positions = [[0,0,Z], [2,0,Z], [0,2,Z],[1.5,1.5,Z],[-0.5,0.5,Z]]
POS_NORM = 1.
initial_positions = [[-1,1,Z], [-1,1.5,Z], [-1,0,Z],[-1,-0.5,Z],[-1,-1,Z]]
# initial_positions = [[0,1,Z], [0,1.5,Z], [0,0,Z],[0,-0.5,Z],[0,-1,Z]]
# initial_positions = [np.concatenate((np.random.uniform(-3,3,2), [Z])) for i in range(N)]
goal_x, goal_y = 3.,3.
# goal_x, goal_y = np.random.uniform(-1,1,2)
obstacle_x, obstacle_y = 1.5, 1.5
obstacle_present = True
static_entities = 1 + (1 if obstacle_present else 0)
logging = False

# swarms_info = []  # Swarms states
# pybullet_info = []  # Pubullet states
# Create Swarms env
# TODO: Get actual env bounds

env = FlockAviary(gui=True, record=True, num_drones=len(initial_positions), act=ActionType.PID, initial_xyzs=np.array(initial_positions), aggregate_phy_steps=int(5))
logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                num_drones=len(initial_positions))
PYB_CLIENT = env.getPyBulletClient()

# Initialize obstacle and goal in the drone env
if obstacle_present:
    p.loadURDF("sphere2.urdf", [obstacle_x, obstacle_y,0.5], globalScaling = 0.5, useFixedBase=1, physicsClientId=PYB_CLIENT)
p.loadURDF("duck_vhacd.urdf", [goal_x, goal_y,0.05],  physicsClientId=PYB_CLIENT)


# Load system edges
edges = swarmnet.utils.system_edges(goals=1, obstacles=1 if obstacle_present else 0, boids=len(initial_positions))
edge_types = swarmnet.utils.one_hot(edges, num_classes=4, dtype=np.float32)
edge_types = np.expand_dims(edge_types, 0)
# print('EDGE TYPES', edge_types, edge_types.shape)

# Load SwarmNet model
model_params = swarmnet.utils.load_model_params(config='/mnt/c/Parth/CS_Projects/drone_swarm/configs/config.json')
model = swarmnet.SwarmNet.build_model(len(initial_positions) + static_entities, 4, model_params, pred_steps=1)
swarmnet.utils.load_model(model, '/mnt/c/Parth/CS_Projects/drone_swarm/models')

start = time.time()
# Initialize action dict, (x,y,z) velocity PID control
action = {i:np.array([0.,0.,0.]) for i in range(len(initial_positions))}
obs, reward, done, info = env.step(action)

drone_states = [np.concatenate((info[agent]['position'][:2]/POS_NORM, info[agent]['velocity'][:2])) for agent in range(len(initial_positions))]
static_entity_state = np.array([[goal_x/POS_NORM, goal_y/POS_NORM, 0, 0]])
if obstacle_present:
    obstacle_state = np.array([[obstacle_x/POS_NORM, obstacle_y/POS_NORM, 0, 0]])
    static_entity_state = np.concatenate((static_entity_state, obstacle_state))
state = np.concatenate((static_entity_state, drone_states))
state = np.expand_dims(state, 0)
state = np.expand_dims(state, 0)
print('STATE', state, 'state shape:', state.shape)


for i in range(12*int(env.SIM_FREQ/env.AGGR_PHY_STEPS)):
    predicted_action = model.predict([state, edge_types])
    # print(predicted_action[:,:,1,:2][0,0]*POS_NORM)
    for agent in range(len(initial_positions)):
        action[agent][:2] = predicted_action[:,:,agent+static_entities,2:][0,0] #*POS_NORM # Swarms posn and vel
    log.append((state, action))
    obs, reward, done, info = env.step(action)
    # print(action)
    # print(i, state, action)
    drone_states = [np.concatenate((info[agent]['position'][:2]/POS_NORM, info[agent]['velocity'][:2])) for agent in range(len(initial_positions))]
    static_entity_state = np.array([[goal_x/POS_NORM, goal_y/POS_NORM, 0, 0]])
    if obstacle_present:
        obstacle_state = np.array([[obstacle_x/POS_NORM, obstacle_y/POS_NORM, 0, 0]])
        static_entity_state = np.concatenate((static_entity_state, obstacle_state))
    state = np.concatenate((static_entity_state, drone_states))
    state = np.expand_dims(state, 0)
    state = np.expand_dims(state, 0)
    if i%env.SIM_FREQ == 0:
        env.render()
    sync(i, start, env.TIMESTEP)
env.close()

# with open('./delta_info.npy', 'wb') as out_file:
#     np.save(out_file, np.array(swarms_info))
#     np.save(out_file, np.array(pybullet_info))
# # logger.save()
# # logger.plot()