import argparse
import random
import time

import gym
import numpy as np
import pybullet as p
import swarmnet
from gym_pybullet_drones.envs.multi_agent_rl.GoalAviary import GoalAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import \
    ActionType
from gym_pybullet_drones.utils.utils import sync

def main():
    # The Z position at which to intitialize the drone
    # When using velocity based PID control, same value of Z
    # needs to be set in Base{Single,Multi}agentAviary.py
    Z = 0.5
    N = ARGS.N
    # Number of entries in the list determines number of drones
    # initial_positions = [[0,0,Z], [2,0,Z], [0,2,Z],[1.5,1.5,Z],[-0.5,0.5,Z]]
    POS_NORM = 1.
    position_list = [[-1,1.5,Z], [-1,1,Z], [-1,0.5,Z], [-1,0,Z], [-1,-0.5,Z], [-1, 2, Z], [-1,-2.5,Z], [-1, 3, Z]]
    initial_positions = position_list[:N]
    # initial_positions = [[0,1,Z], [0,1.5,Z], [0,0,Z],[0,-0.5,Z],[0,-1,Z]]
    # initial_positions = [np.concatenate((np.random.uniform(-3,3,2), [Z])) for i in range(N)]
    # initial_positions = [np.array([ 2.1049795 , -0.96557476,  0.5       ]), np.array([ 1.18638325, -1.1899444 ,  0.5       ]), np.array([0.45437863, 0.45471353, 0.5       ]), np.array([ 0.29974025, -1.05334061,  0.5       ]), np.array([-0.07426064, -2.09950476,  0.5       ])]
    # initial_positions = [np.array([1.45533071, 2.53635661, 0.5       ]), np.array([1.92868378, 2.22490029, 0.5       ]), np.array([0.72355883, 1.95052319, 0.5       ]), np.array([0.98709564, 0.01775948, 0.5       ]), np.array([ 2.33979686, -0.45348238,  0.5       ])]
    # initial_positions = [np.array([1.97552489, 2.52728928, 0.5       ]), np.array([ 1.21273208, -1.48935984,  0.5       ]), np.array([-1.06621393,  2.7865154 ,  0.5       ]), np.array([-1.52576967, -1.0172671 ,  0.5       ]), np.array([-2.6915656 ,  0.75233835,  0.5       ])]
    print(initial_positions)
    NUM_DRONES = len(initial_positions)
    goal_x, goal_y = 3., 3.
    # goal_x, goal_y = np.random.uniform(-1,1,2)
    goal_pos = [goal_x, goal_y, 0.05]
    obstacle_x, obstacle_y = 1., 1.
    obstacle_pos = [(obstacle_x, obstacle_y, Z)]
    obstacle_present = True
    static_entities = 1 + (1 if obstacle_present else 0)
    logging = False
    noise = ARGS.noise
    print(f'[INFO] Noise: {noise}')
    gui=ARGS.gui
    record=ARGS.record

    env = GoalAviary(gui=gui, 
                    record=record,
                    num_drones=len(initial_positions),
                    act=ActionType.PID,
                    initial_xyzs=np.array(initial_positions),
                    aggregate_phy_steps=int(5),
                    goal_pos=goal_pos,
                    obstacle_pos=obstacle_pos,
                    obstacle_present=obstacle_present,
                    noise=noise
                    )

    # Load SwarmNet model
    model_params = swarmnet.utils.load_model_params(config='/mnt/c/Parth/CS_Projects/drone_swarm/configs/config.json')
    model = swarmnet.SwarmNet.build_model(len(initial_positions) + static_entities, 4, model_params, pred_steps=1)
    swarmnet.utils.load_model(model, '/mnt/c/Parth/CS_Projects/drone_swarm/models')

    start = time.time()

    total_trial_reward = 0
    failed_trials = 0
    if ARGS.write_file:
        f = open(ARGS.write_file, "a")
        f.write(f"----Noise: {noise}----\n")
    for trial in range(ARGS.trials):
        # Initialize action dict, (x,y,z) velocity PID control
        action = {i:np.array([0.,0.,0.]) for i in range(NUM_DRONES)}
        obs = env.reset()

        total_reward = 0.
        if not noise:  # If noise is zero then predict action only once
            predicted_action = model.predict([obs[0]['nodes'][np.newaxis, np.newaxis, :], obs[0]['edges'][np.newaxis, :]])

            for agent_idx in range(NUM_DRONES):
                action[agent_idx][:2] = predicted_action[:,:,agent_idx+static_entities,2:][0,0]

        else:
            for agent_idx in range(NUM_DRONES):
                action[agent_idx][:2] = model.predict([obs[agent_idx]['nodes'][np.newaxis, np.newaxis, :], obs[agent_idx]['edges'][np.newaxis, :]])[:,:,agent_idx+static_entities,2:][0,0]
            
        for i in range(12*int(env.SIM_FREQ/env.AGGR_PHY_STEPS)):
            obs, reward, done, info = env.step(action)
            total_reward += sum([reward[agent_idx] for agent_idx in range(N)])
            # Calculate next action
            if not noise:  # If noise is zero then predict action only once
                predicted_action = model.predict([obs[0]['nodes'][np.newaxis, np.newaxis, :], obs[0]['edges'][np.newaxis, :]])
            
                for agent_idx in range(NUM_DRONES):
                    action[agent_idx][:2] = predicted_action[:,:,agent_idx+static_entities,2:][0,0]
            
            else:
                for agent_idx in range(NUM_DRONES):
                    action[agent_idx][:2] = model.predict([obs[agent_idx]['nodes'][np.newaxis, np.newaxis, :], obs[agent_idx]['edges'][np.newaxis, :]])[:,:,agent_idx+static_entities,2:][0,0]

            if i%env.SIM_FREQ == 0 and gui:
                env.render()
            sync(i, start, env.TIMESTEP)
            if done['__any__']:
                print('\n[WARNING] COLLISION')
                failed_trials += 1
                break
                
            print(f'\rTime step: {i} | Reward so far: {total_reward}', end='')

        # Normalize reward 
        total_episode_reward = total_reward/N
        print(f'\nTrial: {trial+1} | Episode reward: {total_episode_reward} | Success rate: {(trial+1) - failed_trials}/{trial+1}')
        total_trial_reward += total_episode_reward
        if ARGS.write_file:
            # Write results to file
            f.write(f"Trial: {trial+1} | Episode Reward: {total_episode_reward}  | Success rate: {(trial+1) - failed_trials}/{trial+1}\n")
    if ARGS.write_file:
        f.write(f"Noise: {ARGS.noise} | Average trial reward: {total_trial_reward/ARGS.trials} | Success rate: {(trial+1) - failed_trials}/{trial+1} ({((trial+1) - failed_trials)/(trial+1)})\n\n")
        f.close()
    env.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=5,
                        help='number of drones')
    parser.add_argument('--record', action='store_true',
                        help='whether to record or not')
    parser.add_argument('--gui', action='store_true',
                        help='whether to show gui or not')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='observation noise')
    parser.add_argument('--write-file', type=str, default="",
                        help='name of write file, if present, results are written to file')
    parser.add_argument('--trials', type=int, default=1,
                        help='number of times to run trials')
    ARGS = parser.parse_args()

    main()

