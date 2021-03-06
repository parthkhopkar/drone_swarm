"""Learning script for multi-agent problems.

Agents are based on `ray[rllib]`'s implementation of PPO and use a custom centralized critic.

Example
-------
To run the script, type in a terminal:

    $ python multiagent.py --num_drones <num_drones> --env <env> --obs <ObservationType> --act <ActionType> --algo <alg> --num_workers <num_workers>

Notes
-----
Check Ray's status at:
    http://127.0.0.1:8265

"""
import argparse
import math
import os
import pdb
import pickle
import subprocess
import time
from datetime import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import ray
# My imports
import swarmnet
import tensorflow as tf
from gym import error, spaces, utils
from gym.spaces import Box, Dict
from gym.utils import seeding
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.GoalAviary import GoalAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType, ObservationType)
from gym_pybullet_drones.utils.Logger import Logger
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTFPolicy, PPOTrainer
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune import register_env
from ray.tune.logger import DEFAULT_LOGGERS
from swarmnet import SwarmNetRL

import shared_constants

N_DIM = 2
OWN_OBS_VEC_SIZE = 6 # Modified at runtime
ACTION_VEC_SIZE = 3 # Modified at runtime
NUM_NODES = 5 + 1 + 1  # TODO Replace with variables
OUTPUT_DIM = 2*N_DIM

#### Useful links ##########################################
# Workflow: github.com/ray-project/ray/blob/master/doc/source/rllib-training.rst
# ENV_STATE example: github.com/ray-project/ray/blob/master/rllib/examples/env/two_step_game.py
# Competing policies example: github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py

############################################################
class CustomTFCentralizedCriticModel(TFModelV2):
    """Multi-agent model that implements a centralized value function.

    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).

    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TFModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        self.action_model = FullyConnectedNetwork(
                                                  Box(low=-1, high=1, shape=(OWN_OBS_VEC_SIZE, )), 
                                                  action_space,
                                                  num_outputs,
                                                  model_config,
                                                  name + "_action"
                                                  )
        self.value_model = FullyConnectedNetwork(
                                                 obs_space, 
                                                 action_space,
                                                 1, 
                                                 model_config, 
                                                 name + "_vf"
                                                 )
        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        self._model_in = [input_dict["obs_flat"], state, seq_lens]
        return self.action_model({"obs": input_dict["obs"]["own_obs"]}, state, seq_lens)

    def value_function(self):
        value_out, _ = self.value_model({"obs": self._model_in[0]}, self._model_in[1], self._model_in[2])
        return tf.reshape(value_out, [-1])
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomTFCentralizedCriticModel, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        layer_1 = tf.keras.layers.Dense(
            128,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(self.inputs)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_1)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        print(input_dict["obs_flat"])
        model_out, self._value_out = self.base_model(input_dict["obs_flat"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}

############################################################
class FillInActions(DefaultCallbacks):
    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0
        action_encoder = ModelCatalog.get_preprocessor_for_space( 
                                                                 Box(-np.inf, np.inf, (ACTION_VEC_SIZE,), np.float32) # Unbounded
                                                                 )
        _, opponent_batch = original_batches[other_id]
        opponent_actions = np.array([action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]])
        to_update[:, -ACTION_VEC_SIZE:] = opponent_actions

############################################################
def central_critic_observer(agent_obs, **kw):
    new_obs = {
        0: {
            "own_obs": agent_obs[0],
            "opponent_obs": agent_obs[1],
            "opponent_action": np.zeros(ACTION_VEC_SIZE), # Filled in by FillInActions
        },
        1: {
            "own_obs": agent_obs[1],
            "opponent_obs": agent_obs[0],
            "opponent_action": np.zeros(ACTION_VEC_SIZE), # Filled in by FillInActions
        },
    }
    return new_obs

############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--num_drones',  default=2,            type=int,                                                                 help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--env',         default='goal',      type=str,             choices=['flock', 'goal'],      help='Help (default: ..)', metavar='')
    parser.add_argument('--obs',         default='kin',        type=ObservationType,                                                     help='Help (default: ..)', metavar='')
    parser.add_argument('--act',         default='pid',  type=ActionType,                                                          help='Help (default: ..)', metavar='')
    parser.add_argument('--algo',        default='cc',         type=str,             choices=['cc'],                                     help='Help (default: ..)', metavar='')
    parser.add_argument('--workers',     default=0,            type=int,                                                                 help='Help (default: ..)', metavar='')
    parser.add_argument('--config',      default='configs/config.json',            type=str,                                             help='Help (default: ..)', metavar='')        
    ARGS = parser.parse_args()

    #### Save directory ########################################
    filename = os.path.dirname(os.path.abspath(__file__))+'/results/save-'+ARGS.env+'-'+str(ARGS.num_drones)+'-'+ARGS.algo+'-'+ARGS.obs.value+'-'+ARGS.act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Uncomment to debug slurm scripts ######################
    # exit()

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Register the custom centralized critic model ##########
    ModelCatalog.register_custom_model("cc_model", SwarmNetRL) # TODO: Register the SwarmNet model in the same way

    #### Register the environment with RLLib##############################
    temp_env_name = "this-aviary-v0"
    register_env(temp_env_name, lambda _: GoalAviary(num_drones=ARGS.num_drones,
                                                                   aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                                   obs=ARGS.obs
                                                                
                                                                   )
                     )

    #### Unused env to extract the act and obs spaces ##########
    temp_env = GoalAviary(num_drones=ARGS.num_drones,
                                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                        obs=ARGS.obs
                                        )
    observer_space = temp_env.observation_space()[0]
    action_space = temp_env.action_space()[0]

    # print(f'Obs space: {observer_space} | Act space: {action_space} | {temp_env.observation_space()}')

    #### Note ##################################################
    # RLlib will create ``num_workers + 1`` copies of the
    # environment since one copy is needed for the driver process.
    # To avoid paying the extra overhead of the driver copy,
    # which is needed to access the env's action and observation spaces,
    # you can defer environment initialization until ``reset()`` is called

    # TODO: Determine the config we need with Tensorflow
    #### Set up the trainer's config ###########################
    config = ppo.DEFAULT_CONFIG.copy() # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    config = {
        "env": temp_env_name,
        "num_workers": 0 + ARGS.workers,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")), # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        "batch_mode": "complete_episodes",
        # "callbacks": FillInActions,
        "framework": "tf"
    }

    #### Set up the model parameters of the trainer's config ###
    model_params = swarmnet.utils.load_model_params(ARGS.config)
    model_params['num_nodes'] = NUM_NODES
    model_params['output_dim'] = OUTPUT_DIM

    config["model"] = { 
        "custom_model": "cc_model",
        "custom_model_config": model_params
    }
    
    #### Set up the multiagent params of the trainer's config ##
    config["multiagent"] = { 
        "policies": {
            "pol0": (None, observer_space, action_space, {})
        },
        "policy_mapping_fn": lambda x: "pol0", # # Function mapping agent ids to policy ids
        # "observation_fn": central_critic_observer, # See rllib/evaluation/observation_function.py for more info
    }

    #### Ray Tune stopping conditions ##########################
    stop = {
        # "timesteps_total": 1,
        # "episode_reward_mean": 0,
        "training_iteration": 1,
    }

    #### Train #################################################
    results = tune.run(
        "PPO",
        stop=stop,
        config=config,
        verbose=True,
        checkpoint_at_end=True,
        local_dir=filename,
    )
    # check_learning_achieved(results, 1.0)

    #### Save agent ############################################
    checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean',
                                                                                   mode='max'
                                                                                   ),
                                                      metric='episode_reward_mean'
                                                      )
    with open(filename+'/checkpoint.txt', 'w+') as f:
        f.write(checkpoints[0][0])

    #### Shut down Ray #########################################
    ray.shutdown()
