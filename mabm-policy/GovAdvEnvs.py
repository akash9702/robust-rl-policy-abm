import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import random

import juliacall
from juliacall import Main as jl

import torch
from copy import deepcopy

from stable_baselines3.common.vec_env import VecNormalize
import matplotlib.pyplot as plt

# register(
#     id='abc-govadv-gov',                                
#     entry_point='GovAdvEnvs:GovABC',
# )

# register(
#     id='abc-govadv-adv',                                
#     entry_point='GovAdvEnvs:AdvABC',
# )

class GovABC(gym.Env):
    def __init__(
        self,
        simulator
    ):
        self.simulator = simulator
        self.action_space = deepcopy(self.simulator.gov_action_space)
        self.observation_space = deepcopy(self.simulator.gov_observation_space)
        self.adv_agent = None

    def reset(
        self,
        seed: int | None = None
    ):
        super().reset(seed = seed)

        # self.reward = 0.0

        return self.simulator.reset_simulator(train_gov = True)
    
    def step(
        self,
        action
    ):
        adv_obs = self.simulator.get_obs_sim(gov = False)
        with torch.no_grad():
            if isinstance(self.adv_agent.env, VecNormalize):
                norm_obs = self.adv_agent.env.normalize_obs(adv_obs)
            else:
                norm_obs = adv_obs
            adv_action, _ = self.adv_agent.predict(norm_obs)
        # obs, rew, term, trun, info = self.simulator.step(action, adv_action, train_gov = True)
        # self.reward += rew
        # if term:
        #     print(f"episode {self.simulator.episode_number} gov reward: {self.reward}")
        # return obs, rew, term, trun, info
        return self.simulator.step(action, adv_action, train_gov = True)


    def render(
        self
    ):
        pass

    def set_adv_agent(
        self,
        adv_agent
    ):
        self.adv_agent = adv_agent


class AdvABC(gym.Env):
    def __init__(
        self,
        simulator
    ):
        self.simulator = simulator
        self.action_space = deepcopy(self.simulator.adv_action_space)
        self.observation_space = deepcopy(self.simulator.adv_observation_space)
        self.gov_agent = None

    def reset(
        self,
        seed: int | None = None
    ):
        super().reset(seed = seed)

        # self.reward = 0.0

        return self.simulator.reset_simulator(train_gov = False)
    
    def step(
        self,
        action
    ):
        gov_obs = self.simulator.get_obs_sim(gov = True)
        with torch.no_grad():
            if isinstance(self.gov_agent.env, VecNormalize):
                norm_obs = self.gov_agent.env.normalize_obs(gov_obs)
            else:
                norm_obs = gov_obs
            gov_action, _ = self.gov_agent.predict(norm_obs)
        # obs, rew, term, trun, info = self.simulator.step(gov_action, action, train_gov = False)
        # self.reward += rew
        # if term:
        #     print(f"episode {self.simulator.episode_number} adv reward: {self.reward}")
        # return obs, rew, term, trun, info
        return self.simulator.step(gov_action, action, train_gov = False)


    def render(
        self
    ):
        pass

    def set_gov_agent(
        self,
        gov_agent
    ):
        self.gov_agent = gov_agent


class ARLGov(gym.Env):
    def __init__(
        self,
        simulator
    ):
        self.simulator = simulator
        self.action_space = deepcopy(self.simulator.gov_action_space)
        self.observation_space = deepcopy(self.simulator.gov_observation_space)
        self.adv_agent = None

    def reset(
        self,
        seed: int | None = None
    ):
        # super().reset(seed = seed)
        
        self.episode_actions = []
        self.episode_rewards = []
        
        obs, info = self.simulator.reset_simulator(train_gov = True)

        # print(f"episode {self.simulator.episode_number} RESET CALLED")

        # adv_obs = np.array([0.0], dtype = np.float32)

        # with torch.no_grad():
        #     if isinstance(self.adv_agent.env, VecNormalize):
        #         norm_obs = self.adv_agent.env.normalize_obs(adv_obs)
        #     else:
        #         norm_obs = adv_obs
        #     adv_action, _ = self.adv_agent.predict(norm_obs)
        
        # self.adv_action = adv_action
        
        # self.simulator.take_adv_action(adv_action)
        

        self.reward = 0.0
        
        return obs, info
    
    def step(
        self,
        action
    ):
        if self.simulator.t == 0:
            adv_obs = np.array([0.0], dtype = np.float32)

            with torch.no_grad():
                if isinstance(self.adv_agent.env, VecNormalize):
                    norm_obs = self.adv_agent.env.normalize_obs(adv_obs)
                else:
                    norm_obs = adv_obs
                adv_action, _ = self.adv_agent.predict(norm_obs)
            
            self.adv_action = adv_action
            
            self.simulator.take_adv_action(adv_action)
        self.episode_actions.append(action)
        obs, rew, term, trun, info = self.simulator.step(action)
        self.reward += rew
        self.episode_rewards.append(rew)

        # if self.simulator.t == self.simulator.T and (self.simulator.episode_number >= 241):
        #     print(f"episode {self.simulator.episode_number} gov reward: {self.reward}")
        #     self.plot_actions_in_episode()
        #     self.reward = 0.0
        return obs, rew, term, trun, info


    def render(
        self
    ):
        pass

    def set_adv_agent(
        self,
        adv_agent
    ):
        self.adv_agent = adv_agent

    def plot_actions_in_episode(
        self
    ):
        plt.plot(np.array(self.episode_actions)[:, 0], label = self.simulator.gov_action_types[0], color = "blue")
        plt.plot(np.array(self.episode_actions)[:, 1], label = self.simulator.gov_action_types[1], color = "green")
        plt.plot(np.array([self.adv_action] * self.simulator.T), label = self.simulator.adv_action_types[0], color = "red")

        plt.ylabel("value")
        plt.xlabel("steps")

        plt.legend()
        plt.title(f"episode {self.simulator.episode_number}")
        plt.show()
        plt.close()

        plt.plot(np.array(self.episode_rewards), label = "gov rewards", color = "blue")
        plt.ylabel("rewards")
        plt.xlabel("steps")
        plt.title(f"episode {self.simulator.episode_number}")
        plt.show()
        plt.close()


class ARLAdv(gym.Env):
    def __init__(
        self,
        simulator
    ):
        self.simulator = simulator
        self.action_space = deepcopy(self.simulator.adv_action_space)
        # self.observation_space = deepcopy(self.simulator.adv_observation_space)
        # self.observation_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(low = 0.0, high = 0.0, shape = (1,), dtype = np.float32)
        self.gov_agent = None

    def reset(
        self,
        seed: int | None = None
    ):
        # super().reset(seed = seed)

        self.episode_actions = []

        obs, info = self.simulator.reset_simulator(train_gov = False)

        adv_obs = np.array([0.0], dtype = np.float32)

        return adv_obs, info
    
    def step(
        self,
        action
    ):
        self.adv_action = action
        # gov_obs, info = self.simulator.reset_simulator(train_gov = True)
        gov_obs = self.simulator.get_obs_sim(gov = True)
        
        self.simulator.take_adv_action(action)
        
        reward = 0.0

        term = False

        self.episode_rewards = []
        
        while not term:
            with torch.no_grad():
                if isinstance(self.gov_agent.env, VecNormalize):
                    norm_obs = self.gov_agent.env.normalize_obs(gov_obs)
                else:
                    norm_obs = gov_obs
                gov_action, _ = self.gov_agent.predict(norm_obs)
            self.episode_actions.append(gov_action)
            gov_obs, rew, term, trun, info = self.simulator.step(gov_action)
            self.episode_rewards.append(-rew)
            reward -= rew
        
        info = self.simulator.get_info_sim()
        terminated = True
        truncated = False
        
        # if self.simulator.episode_number >= 241: 
        #     print(f"episode {self.simulator.episode_number} adv reward: {reward}")
        #     self.plot_actions_in_episode()

        adv_obs = np.array([0.0], dtype = np.float32)

        return adv_obs, reward, terminated, truncated, info

    def render(
        self
    ):
        pass

    def set_gov_agent(
        self,
        gov_agent
    ):
        self.gov_agent = gov_agent

    def plot_actions_in_episode(
        self
    ):
        plt.plot(np.array(self.episode_actions)[:, 0], label = self.simulator.gov_action_types[0], color = "blue")
        plt.plot(np.array(self.episode_actions)[:, 1], label = self.simulator.gov_action_types[1], color = "green")
        plt.plot(np.array([self.adv_action] * self.simulator.T), label = self.simulator.adv_action_types[0], color = "red")

        plt.ylabel("value")
        plt.xlabel("steps")

        plt.legend()
        plt.title(f"episode {self.simulator.episode_number}")
        plt.show()
        plt.close()

        plt.plot(np.array(self.episode_rewards), label = "gov rewards", color = "blue")
        plt.ylabel("rewards")
        plt.xlabel("steps")
        plt.title(f"episode {self.simulator.episode_number}")
        plt.show()
        plt.close()
