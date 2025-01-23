from sirs import SIRS
import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import random
from copy import deepcopy
import dill as pickle
import os
import json

class SIRSGov(gym.Env):


    def __init__(
        self,
        T: int = 100, 
        initial_infected: float = 0.3, 
        N: int = 100,
        base_params: list = [1.5, -2.0, -4.0],
        lockdown_cost_param: float = 0.1,
        reward_type: str = "neg_infected",
        additional: dict = {},
        seed: int = 42
    ):
        self.T = T
        self.initial_infected = initial_infected
        self.N = N
        self.lockdown_cost_param = lockdown_cost_param
        self.reward_type = reward_type
        self.additional = additional

        self.seed = int(seed)

        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)
        

        # initial values of params, gotten through chatGPT (for a covid-like infection)
        self.base_params = base_params
        self.params = torch.Tensor(deepcopy(self.base_params))
        self.original_params = deepcopy(self.params)

        self.t = 0
        self.episode_number = 1
        self.random_shocks = False

        self.states = []

        self.last_action = 0

        self._additional_init()

        self._sirs_init()

        self._create_spaces()

        self.reward_functions = {
            "neg_infected" : self._neg_infected_reward
        }

        self.adv_action_functions = {
            "alpha" : self._adv_alpha,
            "beta" : self._adv_beta,
            "gamma" : self._adv_gamma
        }


    def step(
        self,
        action
    ):
        """
        action is of the type lockdown or no lockdown
        if action = 0, then no lockdown; and = 1, then lockdown
        if locked down, then 
            S -> I probability = 0 
            I -> R probability remains the same
            R -> S probability remains the same
            
        """

        self.t += 1

        self.last_action = action

        if action == 1:
            # set alpha = 0 
            self.params[0] = -np.inf
            pass
        elif action == 0:
            self.params[0] = self.original_params[0]
        else:
            raise ValueError("action value should be 0 or 1")
        
        self.state = self.sirs_model.step(
            params = self.params,
            x = deepcopy(self.state)
        )

        if self.random_shocks:
            self._random_shock()

        self.states.append(self.state)

        obs = self._get_obs()

        reward = self._compute_reward()

        info = self._get_info()
        
        terminated = self._get_terminated()

        truncated = self._get_truncated()

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None
    ):
        super().reset(seed = seed)

        self._sirs_init()

        self.t = 0
        self.random_shocks = False
        
        self.states = []

        self.last_action = 0

        self.params = torch.Tensor(deepcopy(self.base_params))
        self.original_params = deepcopy(self.params)

        self._additional_init()

        return (self._get_obs(), self._get_info())

    def render(
        self        
    ):
        pass

    # SAVING AND LOADING

    # def save(self):

    #     save_dir = os.path.join(self.dir, "saved")
    #     os.makedirs(save_dir, exist_ok = True)

    #     with open(os.path.join(save_dir, "evaluator.pkl"), "wb") as file:
    #         pickle.dump(self, file)
        
    #     file_path = os.path.join(save_dir, "evaluator.pkl")

    #     print(f"RobustnessEvaluation object saved to {file_path}")
        

    # @staticmethod
    # def load(file_path):
    #     with open(file_path, "rb") as file:
    #         obj = pickle.load(file)
        
    #     print(f"RobustnessEvaluation object loaded from {file_path}")
        
    #     return obj
    
    def get_config(
        self
    ):
        config = deepcopy(vars(self))

        keys_to_remove = [
            "t",
            "episode_number",
            "params",
            "original_params",
            "cost",
            "state",
            "states",
            "last_action",
            "reward_functions",
            "action_space",
            "observation_space",
            "sirs_model",
            "random_shocks",
            "_np_random",
            "adv_action_functions"
        ]

        for key in keys_to_remove:
            if key in config:
                del config[key]

        return config
    
    def get_adv_config(
        self,
        adv_action_types,
        gym_spaces_bounds
    ):
        config = deepcopy(vars(self))

        keys_to_remove = [
            "t",
            "episode_number",
            "params",
            "original_params",
            "cost",
            "state",
            "states",
            "last_action",
            "reward_functions",
            "action_space",
            "observation_space",
            "sirs_model",
            "random_shocks",
            "additional",
            "_np_random",
            "adv_action_functions"
        ]

        for key in keys_to_remove:
            if key in config:
                del config[key]

        config["adv_action_types"] = adv_action_types
        config["gym_spaces_bounds"] = gym_spaces_bounds

        return config


    # PRIMARY HELPER FUNCTIONS

    def _additional_init(
        self
    ):
        if "lockdown_cost_param" in self.additional:
            self.lockdown_cost_param = self.additional["lockdown_cost_param"]
        
        if "alpha" in self.additional:
            self.params[0] = self.original_params[0] = self.additional["alpha"]
        if "beta" in self.additional:
            self.params[1] = self.original_params[1] = self.additional["beta"]
        if "gamma" in self.additional:
            self.params[2] = self.original_params[2] = self.additional["gamma"]

        if "random_shocks" in self.additional:
            self.random_shocks = self.additional["random_shocks"]

    def _sirs_init(
        self
    ):
        self.sirs_model = SIRS(
            n_timesteps = self.T,
            i0 = self.initial_infected,
            N = self.N
        )

        self.state = self.sirs_model.initialize(params = self.params)

        self.states.append(self.state)

    def _create_spaces(
        self
    ):
        self.action_space = spaces.Discrete(
            n = 2
        )
        
        self.observation_space = spaces.MultiDiscrete(
            [self.N + 1, self.N + 1, self.N + 1]
        )

    def _get_obs(
        self
    ):
        return np.array(self.state.flatten()[:3])

    def _get_info(
        self
    ):
        return {"state" : self.state[:3].tolist()}
    
    def _get_terminated(
        self
    ):
        terminated = self.t >= self.T
        if terminated:
            self.episode_number += 1
        return terminated
    
    def _get_truncated(
        self
    ):
        return False
    
    def _compute_reward(
        self
    ):
        return self.reward_functions[self.reward_type]()
    
    def _random_shock(
        self
    ):
        """
        randomly put int(N/40) people from S to I
        """
        n_transfers = np.minimum(int(self.N // 40), self.state.flatten()[0])

        self.state[0][0] -= n_transfers
        self.state[0][1] += n_transfers

    # ADV ACTION HELPER FUNCTIONS

    def take_adv_actions(
        self,
        action,
        adv_action_types
    ):
        for action_type, new_value in zip(adv_action_types, action):
            self.adv_action_functions[action_type](new_value)


    def _adv_alpha(
        self,
        new_value
    ):
        self.params[0] = float(new_value)
        self.original_params[0] = float(new_value)

    def _adv_beta(
        self,
        new_value
    ):
        self.params[1] = float(new_value)
        self.original_params[1] = float(new_value)

    def _adv_gamma(
        self,
        new_value
    ):
        self.params[2] = float(new_value)
        self.original_params[2] = float(new_value)

    # REWARD HELPER FUNCTIONS

    def _neg_infected_reward(
        self
    ):
        obs = self._get_obs()
        infected = obs[1]

        cost = 0

        if self.last_action == 1:
            cost = self.lockdown_cost_param * self.N

        return -infected - cost
    
    def _neg_sir_reward(
        self
    ):
        obs = self._get_obs()
        S, I, R = obs

        lockdown_cost = 0
        if self.last_action == 1:
            lockdown_cost = -self.lockdown_cost_param * self.N

        infected_cost = -I

        alpha, beta, gamma = torch.exp(self.params)
        p_si = (1 - torch.exp(-alpha * I / self.N)).item()
        p_ir = (1 - torch.exp(-beta)).item()
        p_rs = (1 - torch.exp(-gamma)).item()

        susceptible_cost = -0.5 * p_si * S
        infected_cost = -I
        recovered_cost = -0.25 * p_rs * p_si * R

        return (susceptible_cost + infected_cost + recovered_cost + lockdown_cost)

        


class RandSIRSGov(gym.Env):

    """
    
    changes the value of rand_param after every episode (note: it doesn't change it during episodes)
    
    """

    _default_gym_spaces_bounds = {
        "act_alpha" : (-1.5, 4.5),
        "act_beta" : (-5.0, 1.0),
        "act_gamma" : (-7.0, -1.0),
        "act_lcp" : (0.00001, 1.0)
    }

    def __init__(
        self,
        T: int = 100, 
        initial_infected: float = 0.1, 
        N: int = 100,
        base_params: list = [1.5, -2.0, -4.0],
        lockdown_cost_param: float = 0.1,
        reward_type: str = "neg_infected",
        rand_params: list = ["beta"],
        additional: dict = {},
        gym_spaces_bounds: dict = {}
    ):
        self.T = T
        self.initial_infected = initial_infected
        self.N = N
        self.lockdown_cost_param = lockdown_cost_param
        self.reward_type = reward_type
        self.additional = additional
        self.rand_params = rand_params

        # initial values of params, gotten through chatGPT (for a covid-like infection)
        self.base_params = base_params
        self.params = torch.Tensor(deepcopy(self.base_params))
        self.original_params = deepcopy(self.params)

        self.t = 0
        self.episode_number = 1
        self.random_shocks = False

        self.states = []

        self.last_action = 0

        self._additional_init()

        self._sirs_init()

        self._create_gym_spaces_bounds(gym_spaces_bounds)

        self._create_spaces()

        self.reward_functions = {
            "neg_infected" : self._neg_infected_reward
        }

        self.rand_functions = {
            "log_uniform" : self._rand_log_uniform,
            "exp_uniform" : self._rand_exp_uniform,
            "log_gauss" : self._rand_log_gauss,
            "exp_gauss" : self._rand_exp_gauss,
            "uniform" : self._rand_uniform,
            "gauss" : self._rand_gauss
        }

        self.change_functions = {
            "alpha" : self._change_alpha,
            "beta" : self._change_beta,
            "gamma" : self._change_gamma,
            "lcp" : self._change_lcp
        }

        self.param_to_rand_type = {
            "alpha" : "uniform",
            "beta" : "gauss",
            "gamma" : "uniform",
            "lcp" : "log_uniform"
        }

        self._randomize()


    def step(
        self,
        action
    ):
        """
        action is of the type lockdown or no lockdown
        if action = 0, then no lockdown; and = 1, then lockdown
        if locked down, then 
            S -> I probability = 0 
            I -> R probability remains the same
            R -> S probability remains the same
            
        """

        self.t += 1

        self.last_action = action

        if action == 1:
            # since "alpha" here is actually log(alpha) in the model, and we want to set alpha = 0
            self.params[0] = -np.inf
            pass
        elif action == 0:
            self.params[0] = self.original_params[0]
        else:
            raise ValueError("action value should be 0 or 1")
        
        self.state = self.sirs_model.step(
            params = self.params,
            x = deepcopy(self.state)
        )

        if self.random_shocks:
            self._random_shock()

        self.states.append(self.state)

        obs = self._get_obs()

        reward = self._compute_reward()

        info = self._get_info()
        
        terminated = self._get_terminated()

        truncated = self._get_truncated()

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None
    ):
        super().reset(seed = seed)

        self.params = torch.Tensor(deepcopy(self.base_params))
        self.original_params = deepcopy(self.params)
        
        self.states = []

        self.last_action = 0

        self._sirs_init()

        self._additional_init()

        self._randomize()

        self.t = 0
        self.random_shocks = False

        return (self._get_obs(), self._get_info())

    def render(
        self        
    ):
        pass

    # BASIC NON-ENV FUNCTIONS

    def get_config(
        self
    ):
        config = deepcopy(vars(self))

        keys_to_remove = [
            "t",
            "episode_number",
            "params",
            "original_params",
            "cost",
            "state",
            "states",
            "last_action",
            "reward_functions",
            "action_space",
            "observation_space",
            "sirs_model",
            "random_shocks",
            "rand_functions",
            "change_functions",
            "param_to_rand_type"
        ]

        for key in keys_to_remove:
            if key in config:
                del config[key]

        return config


    # PRIMARY HELPER FUNCTIONS

    def _additional_init(
        self
    ):
        if "lockdown_cost_param" in self.additional:
            self.lockdown_cost_param = self.additional["lockdown_cost_param"]
        
        if "alpha" in self.additional:
            self.params[0] = self.original_params[0] = self.additional["alpha"]
        if "beta" in self.additional:
            self.params[1] = self.original_params[1] = self.additional["beta"]
        if "gamma" in self.additional:
            self.params[2] = self.original_params[2] = self.additional["gamma"]

        if "random_shocks" in self.additional:
            self.random_shocks = self.additional["random_shocks"]

    def _sirs_init(
        self
    ):
        self.sirs_model = SIRS(
            n_timesteps = self.T,
            i0 = self.initial_infected,
            N = self.N
        )

        self.state = self.sirs_model.initialize(params = self.params)

        self.states.append(self.state)

    def _create_spaces(
        self
    ):
        self.action_space = spaces.Discrete(
            n = 2
        )
        
        self.observation_space = spaces.MultiDiscrete(
            [self.N + 1, self.N + 1, self.N + 1]
        )

    def _create_gym_spaces_bounds(
        self,
        gym_spaces_bounds
    ):
        self.gym_spaces_bounds = {}

        for key in self.rand_params:
            action_key = f"act_{key}"
            if action_key in gym_spaces_bounds:
                self.gym_spaces_bounds[action_key] = gym_spaces_bounds[action_key]
            else:
                self.gym_spaces_bounds[action_key] = self._default_gym_spaces_bounds[action_key]

    def _get_obs(
        self
    ):
        return np.array(self.state.flatten()[:3])

    def _get_info(
        self
    ):
        return {"state" : self.state[:3].tolist()}
    
    def _get_terminated(
        self
    ):
        terminated = self.t >= self.T
        if terminated:
            self.episode_number += 1
        return terminated
    
    def _get_truncated(
        self
    ):
        return False
    
    def _compute_reward(
        self
    ):
        return self.reward_functions[self.reward_type]()
    
    def _random_shock(
        self
    ):
        """
        randomly put int(N/40) people from S to I
        """
        n_transfers = np.minimum(int(self.N // 40), self.state.flatten()[0])

        self.state[0][0] -= n_transfers
        self.state[0][1] += n_transfers

    # RANDOMIZATION HELPER FUNCTIONS

    def _randomize(
        self
    ):
        for param in self.rand_params:

            rand_type = self.param_to_rand_type[param]

            new_value = self.rand_functions[rand_type](
                low = self._default_gym_spaces_bounds[f"act_{param}"][0],
                high = self._default_gym_spaces_bounds[f"act_{param}"][1]
            )

            self.change_functions[param](new_value)

    def _change_alpha(
        self,
        new_value
    ):
        self.params[0] = self.original_params[0] = new_value

    def _change_beta(
        self,
        new_value
    ):
        self.params[1] = self.original_params[1] = new_value

    def _change_gamma(
        self,
        new_value
    ):
        self.params[2] = self.original_params[2] = new_value

    def _change_lcp(
        self,
        new_value
    ):
        self.lockdown_cost_param = new_value

    def _rand_uniform(
        self,
        low,
        high
    ):
        return random.uniform(low, high)
    
    def _rand_gauss(
        self,
        low,
        high
    ):
        return random.gauss((low + high) / 2, (high - low) / 15)

    def _rand_exp_uniform(
        self,
        low,
        high
    ):
        return np.log(random.uniform(np.exp(low), np.exp(high)))
    
    def _rand_log_uniform(
        self,
        low,
        high
    ):
        return np.exp(random.uniform(np.log(low), np.log(high)))
    
    def _rand_log_gauss(
        self,
        low,
        high
    ):
        return np.exp(random.gauss(
            (np.log(low) + np.log(high)) / 2,
            (np.log(high) - np.log(low)) / 5
        ))
    
    def _rand_exp_gauss(
        self,
        low,
        high
    ):
        pass
    

    # REWARD HELPER FUNCTIONS

    def _neg_infected_reward(
        self
    ):
        obs = self._get_obs()
        infected = obs[1]

        cost = 0

        if self.last_action == 1:
            cost = self.lockdown_cost_param * self.N

        return -infected - cost
    
    def _neg_sir_reward(
        self
    ):
        obs = self._get_obs()
        S, I, R = obs

        lockdown_cost = 0
        if self.last_action == 1:
            lockdown_cost = -self.lockdown_cost_param * self.N

        infected_cost = -I

        alpha, beta, gamma = torch.exp(self.params)
        p_si = (1 - torch.exp(-alpha * I / self.N)).item()
        p_ir = (1 - torch.exp(-beta)).item()
        p_rs = (1 - torch.exp(-gamma)).item()

        susceptible_cost = -0.5 * p_si * S
        infected_cost = -I
        recovered_cost = -0.25 * p_rs * p_si * R

        return (susceptible_cost + infected_cost + recovered_cost + lockdown_cost)
    
class RARLSIRSGov(gym.Env):
    def __init__(
        self,
        simulator
    ):
        self.simulator = simulator

        self.action_space = deepcopy(self.simulator.gov_action_space)
        self.observation_space = deepcopy(self.simulator.observation_space)
        
        self.adv_agent = None

    def step(
        self,
        action
    ):
        obs = self.simulator.get_obs_sim()

        with torch.no_grad():
            adv_action, _ = self.adv_agent.predict(obs)
        
        return self.simulator.step(gov_action = action, adv_action = adv_action, train_gov = True)

    def reset(
        self,
        seed: int | None = None
    ):
        super().reset(seed = seed)

        return self.simulator.reset()


    def render(
        self
    ):
        pass

    def set_adv_agent(
        self,
        adv_agent
    ):
        self.adv_agent = adv_agent


class RARLSIRSAdv(gym.Env):
    def __init__(
        self,
        simulator
    ):
        self.simulator = simulator

        self.action_space = deepcopy(self.simulator.adv_action_space)
        self.observation_space = deepcopy(self.simulator.observation_space)
        
        self.gov_agent = None

    def step(
        self,
        action
    ):
        obs = self.simulator.get_obs_sim()

        with torch.no_grad():
            gov_action, _ = self.gov_agent.predict(obs)
        
        return self.simulator.step(gov_action = gov_action, adv_action = action, train_gov = False)

    def reset(
        self,
        seed: int | None = None
    ):
        super().reset(seed = seed)

        return self.simulator.reset()


    def render(
        self
    ):
        pass

    def set_gov_agent(
        self,
        gov_agent
    ):
        self.gov_agent = gov_agent

class RARLSIRSGov2(gym.Env):
    def __init__(
        self,
        simulator
    ):
        self.simulator = simulator

        self.action_space = deepcopy(self.simulator.gov_action_space)
        self.observation_space = deepcopy(self.simulator.observation_space)
        
        self.adv_agent = None

    def step(
        self,
        action
    ):
        return self.simulator.step(gov_action = action)

    def reset(
        self,
        seed: int | None = None
    ):
        super().reset(seed = seed)

        obs, info = self.simulator.reset()

        if self.adv_agent != None:
            adv_obs = 0
            with torch.no_grad():
                adv_action, _ = self.adv_agent.predict(adv_obs)
        else:
            adv_action = self.simulator.adv_action_space.sample()

        self.simulator.take_adv_action(adv_action)

        return obs, info


    def render(
        self
    ):
        pass

    def set_adv_agent(
        self,
        adv_agent
    ):
        self.adv_agent = adv_agent


class RARLSIRSAdv2(gym.Env):
    def __init__(
        self,
        simulator
    ):
        self.simulator = simulator

        self.action_space = deepcopy(self.simulator.adv_action_space)
        # self.observation_space = deepcopy(self.simulator.observation_space)
        self.observation_space = spaces.Discrete(1)
        
        self.gov_agent = None

    def step(
        self,
        action
    ):
        # reset the simulator while also getting obs for gov_action
        obs, info = self.simulator.reset()

        # take adv action in simulator before episode starts
        self.simulator.take_adv_action(action)

        # episode run and reward computation
        term = False
        reward = 0.0
        while not term:
            with torch.no_grad():
                gov_action, _ = self.gov_agent.predict(obs)
            obs, rew, term, trun, info = self.simulator.step(gov_action)
            reward -= rew
        
        obs, info = self.simulator.reset()
        terminated = True
        truncated = False

        adv_obs = 0

        return adv_obs, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None
    ):
        super().reset(seed = seed)
        obs, info = self.simulator.reset()
        adv_obs = 0
        return adv_obs, info


    def render(
        self
    ):
        pass

    def set_gov_agent(
        self,
        gov_agent
    ):
        self.gov_agent = gov_agent