import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import random

import juliacall
from juliacall import Main as jl

from typing import List, Any, Dict, Tuple, Callable

import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from gymnasium.utils import seeding

# register(
#     id='abc-vanilla-gov',                                
#     entry_point='VanillaGov:VanillaGovABC',
# )

class VanillaGovABC(gym.Env):

    # _default_gym_spaces_bounds: Dict = {
    #     "act_income_tax": (0.0, 1.0),
    #     # "obs_gdp": (0, np.inf),
    #     # "obs_gdp": (0.0, 1.0),
    #     "obs_gdp": (0, 1e20),
    #     "obs_gdp_deficit": (-np.inf, np.inf),
    #     "obs_consumption": (-np.inf, np.inf),
    #     "obs_gov_bonds": (-np.inf, np.inf),
    #     "obs_bank_reserves": (-np.inf, np.inf),
    #     "obs_bank_deposits": (-np.inf, np.inf),
    #     "obs_investment": (-np.inf, np.inf),
    #     "obs_bank_profits": (-np.inf, np.inf),
    #     "obs_bank_dividends": (-np.inf, np.inf),
    #     "obs_bank_equity": (-np.inf, np.inf),
    #     "obs_gov_balance": (-np.inf, np.inf),
    #     "obs_unemployment_rate": (0.0, 1.0),
    #     "obs_inflation_rate": (-np.inf, np.inf),
    #     "obs_bankruptcy_rate": (-np.inf, np.inf),
    #     "act_subsidy": (0.0, 1.0)
    # }

    _default_gym_spaces_bounds: Dict = {
        "act_income_tax": (0.0, 1.0),
        "act_interest_rate": (0.0, 1.0),
        "obs_gdp": (0, 1e20),
        "obs_gdp_deficit": (-1e20, 1e20),
        "obs_consumption": (0, 1e20),
        "obs_gov_bonds": (-1e20, 1e20),
        "obs_bank_reserves": (-1e20, 1e20),
        "obs_bank_deposits": (-1e20, 1e20),
        "obs_investment": (-1e20, 1e20),
        "obs_bank_profits": (-1e20, 1e20),
        "obs_bank_dividends": (-1e20, 1e20),
        "obs_bank_equity": (-1e20, 1e20),
        "obs_gov_balance": (-1e20, 1e20),
        "obs_unemployment_rate": (0.0, 1.0),
        "obs_inflation_rate": (-100, 100),
        "obs_bankruptcy_rate": (-1,1),
        "act_subsidy": (0.0, 1.0)
    }

    def __init__(
        self,
        T: int = 300,
        W: int = 250,
        F: int = 25,
        N: int = 5,
        t_burnin: int = 30,
        n_const_steps: int = 1,
        gym_spaces_bounds: Dict[str, Tuple[float, float]] = {},
        action_types: List[str] = ["income_tax", "subsidy"],
        reward_type: str = "gdp_growth_stability",
        observation_types: List[str] = ["gdp", "bank_deposits"],
        n_prev_obs: int = 2,
        params: str = "PARAMS_ORIGINAL",
        plot: bool = False,
        plot_after_every: int = 1,
        additional: Any = {"alpha": 100.0},
        seed: int = 42
    ):
        self.T = T
        self.W = W
        self.F = F
        self.N = N
        self.t_burnin = t_burnin
        self.n_const_steps = n_const_steps
        self.n_prev_obs = n_prev_obs
        self.plot = plot
        self.plot_after_every = plot_after_every
        self.additional = additional
        self.seed = seed
        
        # note that t and step_number aren't the same things since 1 ABCredit step != 1 env step (since 1 env step might be, for instance, 4 ABC steps)
        self.t = 0
        self.step_number = 0
        self.episode_number = 0

        self.allowed_action_types = [
            "income_tax",
            "subsidy",
            "interest_rate"
        ]

        self.actions_to_properties = {
            "income_tax" : "tax_rate",
            "subsidy" : "subsidy",
            "interest_rate" : "interest_rate",
            "firm_invest_prob" : "Iprob",
            "consumption_wealth_ratio" : "chi",
            "memory_parameter" : "xi",
            "labour_prod" : "alpha",
            "capital_prod" : "k"
        }

        self.allowed_observation_types = [
            "gdp",
            "gdp_deficit",
            "consumption",
            "gov_bonds",
            "bank_reserves",
            "bank_deposits",
            "investment",
            "bank_profits",
            "bank_dividends",
            "bank_equity",
            "gov_balance",
            "unemployment_rate",
            "inflation_rate",
            "bankruptcy_rate"
        ]

        self.observations_to_properties = {
            "gdp" : "Y_real",
            "gdp_deficit": "deficitGDP",
            "consumption": "consumption",
            "gov_bonds": "bonds",
            "bank_reserves": "reserves",
            "bank_deposits": "deposits",
            "investment": "Investment",
            "bank_profits": "profitsB",
            "bank_dividends": "dividendsB",
            "bank_equity": "E",
            "gov_balance": "GB",
            "unemployment_rate": "Un",
            "inflation_rate": "inflationRate",
            "bankruptcy_rate": "bankruptcy_rate"
        }

        self.allowed_params = [
            "PARAMS_ORIGINAL"
        ]

        self.allowed_reward_types = [
            "gdp_growth_absolute",
            "gdp_growth_percentage",
            "gdp_growth_deficit",
            "gdp_growth_stability",
            "gdp_growth_deficit_stability",
            "gdp_growth_stability_squared",
            "gdp_stability_fixed"
        ]

        # dictionary storing different reward functions 
        self.reward_functions = {
            "gdp_growth_absolute" : self._gdp_growth_absolute_reward,
            "gdp_growth_percentage" : self._gdp_growth_percentage_reward,
            "gdp_growth_deficit" :  self._gdp_growth_deficit_reward,
            "gdp_growth_stability" : self._gdp_growth_stability_reward,
            "gdp_growth_deficit_stability" : self._gdp_growth_deficit_stability_reward,
            "gdp_growth_stability_squared" : self._gdp_growth_stability_squared_reward,
            "gdp_stability_fixed" : self._gdp_stability_fixed
        }

        if not self._check_proper_reward(reward_type):
            raise ValueError("reward should be one of " + str(self.allowed_reward_types))
        if not self._check_proper_params(params):
            raise ValueError("params should be one of " + str(self.allowed_params))
        if not self._check_proper_actions(action_types):
            raise ValueError("action_types should be one of " + str(self.allowed_action_types))
        if not self._check_proper_observations(observation_types):
            raise ValueError("observation_types should be one of " + str(self.allowed_observation_types))
        
        # seeding gym
        # seeding.np_random(self.seed)

        # import ABCredit and initialise the model through Julia
        # jl.seval("using Pkg")
        
        # jl.seval("Pkg.add(\"ABCredit\")")
        # jl.seval("Pkg.instantiate()")
        jl.seval("using ABCredit")
        jl.seval("using Random")
        jl.seval("Random.seed!")(self.seed)
        self._julia_model_init()

        
        self._create_gym_spaces_bounds(gym_spaces_bounds)
        self._create_spaces()

        # self.state = self._get_obs().item()

    def step(
        self,
        action
    ):
        self.step_number += 1
        # print(f"\nstep {self.step_number} started")

        action_np = np.array(action, dtype = np.float32)
        for i in range(action_np.shape[0]):
            self.model.params[jl.Symbol(self.actions_to_properties[self.action_types[i]])] = action_np[i]
            # print(f"setting {self.actions_to_properties[self.action_types[i]]}'s value to {action_np[i]}")

        for i in range(self.n_const_steps):
            self.t += 1
            # print(f"model step {self.t}")
            jl.seval("one_model_step!")(self.model)
            jl.seval("ABCredit.update_data!")(self.data, self.model)

            if self.t >= self.T:
                break

        reward = self._get_reward()
        obs = self._get_obs()
        terminated = self._get_terminated()
        info = self._get_info()
        truncated = self._get_truncated()

        # print(f"reward is {reward}")
        # print(f"obs is: \n{obs}\n")
        
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None
    ):
        super().reset(seed = seed)

        self.t = 0
        self.step_number = 0

        self._julia_model_init()

        return self._get_obs(), self._get_info()

    def render(
        self
    ):
        pass

    def get_config(
        self
    ):
        config = deepcopy(vars(self))

        keys_to_remove = [
            "allowed_action_types", 
            "allowed_observation_types", 
            "actions_to_properties", 
            "t", 
            "step_number", 
            "episode_number", 
            "observations_to_properties", 
            "allowed_params", 
            "allowed_reward_types", 
            "reward_functions", 
            "params", 
            "data", 
            "model", 
            "action_space", 
            "observation_space"
        ]
        
        for key in keys_to_remove:
            if key in config:
                del config[key]

        return config
    
    def get_adv_config(
        self,
        adv_action_types: list = ["consumption_wealth_ratio"],
        gym_spaces_bounds: list = {"act_consumption_wealth_ratio" : (0.0, 1.0)}
    ):
        config = deepcopy(vars(self))

        # print(f"\nprinting gsb inside get_adv_config")
        # print(gym_spaces_bounds)

        keys_to_remove = [
            "allowed_action_types", 
            "allowed_observation_types", 
            "actions_to_properties", 
            "t", 
            "step_number", 
            "episode_number", 
            "observations_to_properties", 
            "allowed_params", 
            "allowed_reward_types", 
            "reward_functions", 
            "params", 
            "data", 
            "model", 
            "action_space", 
            "observation_space",
            "_np_random"
        ]
        
        for key in keys_to_remove:
            if key in config:
                del config[key]

        config["gov_observation_types"] = config["adv_observation_types"] = config["observation_types"]
        del config["observation_types"]
        config["gov_action_types"] = config["action_types"]
        del config["action_types"]
        config["adv_action_types"] = adv_action_types

        for param in gym_spaces_bounds:
            config["gym_spaces_bounds"][param] = gym_spaces_bounds[param]

        # for param in config["gym_spaces_bounds"]:
        #     if param in gym_spaces_bounds:
        #         config["gym_spaces_bounds"][param] = gym_spaces_bounds[param]

        return config
    
    def get_default_action(
        self
    ):
        action = self.action_space.sample()

        params_jl = deepcopy(jl.seval(str("ABCredit." + self.params)))
        
        for i in range(len(self.action_types)):
            action_param = self.action_types[i]
            action[i] = params_jl[jl.Symbol(self.actions_to_properties[action_param])]

        return action


    def _get_reward(
        self
    ):
        return self.reward_functions[self.reward_type]()

    def _get_terminated(
        self
    ):
        terminated = self.t >= self.T

        if terminated:
            
            if self.episode_number % self.plot_after_every == 0 and self.plot:
                self._plot_env()
            
            
            self.episode_number += 1

        return terminated

    def _get_truncated(
        self
    ):
        return self.t >= self.T

    def _get_obs(
        self
    ):
        
        max_gdp = 950 * self.W / 500
        current_time = self.model.agg.timestep - 1

        n_observation_types = len(self.observation_types)
        obs = np.zeros(shape = (n_observation_types, self.n_prev_obs), dtype = np.float32)

        n_real_observations = min(self.n_prev_obs, current_time)
        obs_no = 0
        for observation in self.observation_types:
            for iter in range(n_real_observations):
                attribute_value = getattr(self.data, self.observations_to_properties[observation])[current_time - n_real_observations + iter + 1]
                obs[obs_no][self.n_prev_obs - n_real_observations + iter] = attribute_value

        

            # if some observations are still 0 then set them to the value of the last known observation
            if n_real_observations > 0:
                obs[obs_no][0:self.n_prev_obs - n_real_observations] = obs[obs_no][self.n_prev_obs - n_real_observations]
            
            # if observation == "gdp":
            #     obs[obs_no] /= max_gdp

            # for iter in range(self.n_prev_obs - n_real_observations):
            #     obs[obs_no][iter] = obs[obs_no][self.n_prev_obs - n_real_observations]
            
            obs_no += 1


        return obs

        # n_observations = len(self.observation_types)
        # obs = np.zeros(shape = (n_observations, ), dtype = np.float32)

        # obs_no = 0
        # for observation in self.observation_types:
        #     attribute_value = getattr(self.data, self.observations_to_properties[observation])[current_time]
        #     obs[obs_no] = attribute_value
        #     obs_no += 1

        # return obs

    # TODO: write this properly
    def _get_info(
        self
    ):
        return {}
        


    def _create_gym_spaces_bounds(
        self,
        gym_spaces_bounds
    ):
        self.gym_spaces_bounds = {}

        for action in self.action_types:
            action_key = str("act_" + action)
            if action_key not in gym_spaces_bounds:
                self.gym_spaces_bounds[action_key] = self._default_gym_spaces_bounds[action_key]
            else:
                self.gym_spaces_bounds[action_key] = gym_spaces_bounds[action_key]
        
        for observation in self.observation_types:
            observation_key = str("obs_" + observation)
            if observation_key not in gym_spaces_bounds:
                self.gym_spaces_bounds[observation_key] = self._default_gym_spaces_bounds[observation_key]
            else:
                self.gym_spaces_bounds[observation_key] = gym_spaces_bounds[observation_key]
        
    
    
    def _plot_env(
        self
    ):
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

        axes[0,0].plot(self.data.Y_real)
        axes[0,0].set_title("gdp")

        axes[0,1].plot(self.data.inflationRate)
        axes[0,1].set_title("inflation")

        axes[0,2].plot(self.data.Un)
        axes[0,2].set_title("unemployment")

        axes[1,0].plot(self.data.consumption)
        axes[1,0].set_title("consumption")

        axes[1,1].plot(self.data.totalDeb)
        axes[1,1].set_title("gross debt")

        axes[1,2].plot(self.data.Investment)
        axes[1,2].set_title("gross investment")

        axes[2,0].plot(self.data.totK)
        axes[2,0].set_title("capital stock")

        axes[2,1].plot(self.data.profitsB)
        axes[2,1].set_title("bank profits")

        axes[2,2].plot(self.data.E)
        axes[2,2].set_title("bank equity")

        plt.tight_layout()
        plt.show()
        plt.close()

        # jl.seval("Pkg.add(\"Plots\")")
        # jl.seval("using Plots")

        # p1 = jl.seval("plot")(self.data.Y_real, title = "gdp", titlefont = 10)
        # p2 = jl.seval("plot")(self.data.inflationRate, title = "inflation", titlefont = 10)
        # p3 = jl.seval("plot")(self.data.Un, title = "unemployment", titlefont = 10)
        # p4 = jl.seval("plot")(self.data.consumption, title = "consumption", titlefont = 10)
        # p5 = jl.seval("plot")(self.data.totalDeb, title = "gross debt", titlefont = 10)
        # p6 = jl.seval("plot")(self.data.Investment, title = "gross investment", titlefont = 10)
        # p7 = jl.seval("plot")(self.data.totK, title = "capital stock", titlefont = 10)
        # p8 = jl.seval("plot")(self.data.profitsB, title = "bank profits", titlefont = 10)
        # p9 = jl.seval("plot")(self.data.E, title = "bank equity", titlefont = 10)

        # p = jl.seval("plot")(p1, p2, p3, p4, p5, p6, p7, p8, p9, layout = (3, 3), legend = False)
        # jl.seval("display")(p)

    def _julia_model_init(
        self
    ):
        # making (for instance) ABCredit.PARAMS_ORIGINAL as a Julia object
        params_jl = deepcopy(jl.seval(str("ABCredit." + self.params)))

        # setting random seed to self.seed (since that's the default in the ABCredit code)
        # jl.seval("Random.seed!")(self.seed)

        # change behavioural parameters before model creation
        # print("\nchanging behavioural param values")
        if "change" in self.additional and self.additional["change"]:
            for param, new_value in self.additional["params_to_change"].items():
                params_jl[jl.Symbol(self.actions_to_properties[param])] = new_value
                # print(f"{self.actions_to_properties[param]}'s value changed to {new_value}")
        # print()
        # creating the model in julia
        self.model = jl.seval("initialise_model")(self.W, self.F, self.N, params_jl)

        # burn_in
        for i in range(self.t_burnin):
            jl.seval("one_model_step!")(self.model)

        # reset time counter
        self.model.agg.timestep = 1

        # initializing data collector
        self.data = jl.seval("ABCreditData")(self.T)
        # store variables at initialisation
        jl.seval("ABCredit.update_data!")(self.data, self.model)

    def _check_proper_actions(
            self, 
            action_types
    ) -> bool:
        """
        checks that the action_types given by the user are proper actions 
        """

        if isinstance(action_types, List):
            if set(action_types).issubset(set(self.allowed_action_types)):
                self.action_types = action_types
                return True
            else:
                return False
            
    def _check_proper_observations(
        self,
        observation_types
    ):
        """
        checks that the observation_types given by the user are proper observations
        """

        if isinstance(observation_types, List):
            if set(observation_types).issubset(set(self.allowed_observation_types)):
                self.observation_types = observation_types
                return True
            else:
                return False
            
    def _check_proper_reward(
            self, 
            reward_type
    ) -> bool:
        """
        checks that the reward_type given by the user is a proper reward type 
        """
        
        if reward_type in self.allowed_reward_types:
            self.reward_type = reward_type
            return True
        else:
            return False

    def _check_proper_params(
        self,
        params: str
    ):
        if params in self.allowed_params:
            self.params = params
            return True
        else:
            return False
        
    def _create_spaces(
            self
    ) -> None:
        """
        creates the action_space and the observation_space of the gym environment
        """

        # self.gym_spaces_bounds = gym_spaces_bounds

        # if gym_spaces_bounds == None:
        #     self._create_gym_spaces_bounds()

        

        number_of_actions = int(len(self.action_types))

        num_actions_in_bounds_dict = 0
        for variable, space_range in self.gym_spaces_bounds.items():
            if variable[: 3] == "act":
                num_actions_in_bounds_dict += 1
        
        if not num_actions_in_bounds_dict == number_of_actions:
            raise ValueError("Number of action spaces in gym spaces bounds doesn't match number of action types.")

        actions_low_array = np.zeros(shape = (number_of_actions, ), dtype = np.float32)
        actions_high_array = np.zeros(shape = (number_of_actions, ), dtype = np.float32)

        # TODO: put a check on the face that the names in gym_spaces_bounds should match action and observation names
        for i in range(number_of_actions):
            action = self.action_types[i]
            actions_low_array[i] = self.gym_spaces_bounds[str("act_" + action)][0]
            actions_high_array[i] = self.gym_spaces_bounds[str("act_" + action)][1]
        
        self.action_space = spaces.Box(
            low = actions_low_array,
            high = actions_high_array,
            shape = (number_of_actions, ),
            dtype = np.float32
        )

        n_observation_types = len(self.observation_types)
        observations_low_array = np.zeros(shape = (n_observation_types, self.n_prev_obs), dtype = np.float32)
        observations_high_array = np.zeros(shape = (n_observation_types, self.n_prev_obs), dtype = np.float32)

        for i in range(n_observation_types):
            observation = self.observation_types[i]
            for j in range(self.n_prev_obs):
                observations_low_array[i][j] = self.gym_spaces_bounds[str("obs_" + observation)][0]
                observations_high_array[i][j] = self.gym_spaces_bounds[str("obs_" + observation)][1]

        self.observation_space = spaces.Box(
            low = observations_low_array,
            high = observations_high_array,
            shape = (n_observation_types, self.n_prev_obs),
            dtype = np.float32
        )

        # number_of_observations = len(self.observation_types)

        # observations_low_array = np.zeros(shape = (number_of_observations, ), dtype = np.float32)
        # observations_high_array = np.zeros(shape = (number_of_observations, ), dtype = np.float32)

        # for i in range(number_of_observations):
        #     observation = self.observation_types[i]
        #     observations_low_array[i] = self.gym_spaces_bounds[str("obs_" + observation)][0]
        #     observations_high_array[i] = self.gym_spaces_bounds[str("obs_" + observation)][1]

        # self.observation_space = spaces.Box(
        #     low = observations_low_array,
        #     high = observations_high_array,
        #     shape = (number_of_observations, ),
        #     dtype = np.float32
        # )

    def _gdp_growth_absolute_reward(
        self
    ):
        current_time = self.model.agg.timestep - 1
        last_time = current_time - self.n_const_steps 

        current_gdp = getattr(self.data, self.observations_to_properties["gdp"])[current_time]
        last_gdp = getattr(self.data, self.observations_to_properties["gdp"])[last_time]

        return current_gdp

        return current_gdp - last_gdp

    def _gdp_growth_percentage_reward(
        self
    ):
        """
        returns the fractional growth in gdp
        """

        current_time = self.model.agg.timestep - 1
        last_time = current_time - self.n_const_steps

        current_gdp = getattr(self.data, self.observations_to_properties["gdp"])[current_time]
        last_gdp = getattr(self.data, self.observations_to_properties["gdp"])[last_time]

        return current_gdp / last_gdp - 1
    
    def _gdp_growth_deficit_reward(
        self
    ):
        """
        returns a function of the gdp growth and gdp deficit
        assumes that additional has a parameter called "alpha" which signifies the relative importance of growth and deficit

        self.data.deficitGDP is the ratio of total deficit to the current gdp
        """

        alpha = self.additional["alpha"]

        current_time = self.model.agg.timestep - 1
        last_time = current_time - self.n_const_steps

        current_gdp = getattr(self.data, self.observations_to_properties["gdp"])[current_time]
        last_gdp = getattr(self.data, self.observations_to_properties["gdp"])[last_time]

        current_deficit_ratio = getattr(self.data, self.observations_to_properties["gdp_deficit"])[current_time]
        last_deficit_ratio = getattr(self.data, self.observations_to_properties["gdp_deficit"])[last_time]

        current_deficit = current_deficit_ratio * current_gdp
        last_deficit = last_deficit_ratio * last_gdp

        # return current_gdp - alpha * current_deficit

        return (current_gdp - last_gdp) - alpha * (current_deficit - last_deficit)

        # max_possible_gdp = self.model.params[jl.Symbol("alpha")] * self.W

        # current_gdp / max_possible_gdp - alpha * max(current_deficit_ratio, 0) [govt doesn't want surplus]
        # (max_gdp - current)^2 instead of ratio

    def _gdp_growth_stability_reward(
        self
    ):
        """
        assumes that self.additional has two parameters:
            - alpha: indicates the balance between growth and stability
            - n_prev (not used right now): indicates the number of previous observations to take in the notion of stability
        """

        alpha = self.additional["alpha"]

        # max gdp in real terms: producivity * number of workers * initial price        
        max_gdp =  0.66667 * self.W * 3.0 #1000  / 500

        current_time = self.model.agg.timestep - 1
        current_gdp = getattr(self.data, self.observations_to_properties["gdp"])[current_time]
        current_deficit_ratio = getattr(self.data, self.observations_to_properties["gdp_deficit"])[current_time]
        
        last_time = current_time - self.n_const_steps
        last_gdp = getattr(self.data, self.observations_to_properties["gdp"])[last_time]
        # return (current_gdp - last_gdp) - alpha * np.abs(current_gdp - last_gdp)
        # return  + (current_gdp / max_gdp) - alpha * np.abs(current_gdp - last_gdp)**2         
        # if current_deficit_ratio < 0.0:
        #     return current_gdp / max_gdp
        # else:
        #     return current_gdp / max_gdp - alpha * current_deficit_ratio
        
        gdp_crash = max(0, last_gdp - current_gdp)
        
        return current_gdp / max_gdp - alpha * (gdp_crash/max_gdp)**2
    

    def _gdp_growth_deficit_stability_reward(
        self
    ):
        """
        assumes that self.additional has two parameters:
            - alpha: indicates the weight given to stability (should be around 0.1-1)
            - beta: indicates the weight given to deficit (should be around 0.001-0.01)
            - n_prev (not used right now): indicates the number of previous observations to take in the notion of stability
        """

        alpha = self.additional["alpha"]
        beta = self.additional["beta"]

        current_time = self.model.agg.timestep - 1
        last_time = current_time - self.n_const_steps

        current_gdp = getattr(self.data, self.observations_to_properties["gdp"])[current_time]
        last_gdp = getattr(self.data, self.observations_to_properties["gdp"])[last_time]

        current_deficit_ratio = getattr(self.data, self.observations_to_properties["gdp_deficit"])[current_time]
        last_deficit_ratio = getattr(self.data, self.observations_to_properties["gdp_deficit"])[last_time]

        current_deficit = current_deficit_ratio * current_gdp
        last_deficit = last_deficit_ratio * last_gdp

        return (current_gdp - last_gdp) - alpha * np.abs(current_gdp - last_gdp) - beta * (current_deficit - last_deficit)
    
    def _gdp_growth_stability_squared_reward(
        self
    ):
        """
        here, the measure of stability will take account of the last (n_prev_obs - 1) observations and check what is the squared difference of the current observation from their average
        """
        current_time = self.model.agg.timestep - 1
        last_time = current_time - self.n_const_steps

        current_gdp = getattr(self.data, self.observations_to_properties["gdp"])[current_time]
        last_gdp = getattr(self.data, self.observations_to_properties["gdp"])[last_time]

        obs = self._get_obs()
        
        obs_gdp_iter = -1
        for i in range(len(self.observation_types)):
            if self.observation_types[i] == "gdp":
                obs_gdp_iter = i
                break

        # the numerator is the sum of the first (n_prev_obs - 1) elements of gdp; the denominator is n_prev_obs - 1
        average_past_gdp = np.sum(obs[obs_gdp_iter][0 : -1]) / (obs.shape[-1] - 1)
        
        alpha = self.additional["alpha"]

        return (current_gdp - last_gdp) - alpha * (current_gdp - average_past_gdp) ** 2
    
    def _gdp_stability_fixed(self):
        
        value = self.additional["value"]
        current_time = self.model.agg.timestep - 1
        current_gdp = getattr(self.data, self.observations_to_properties["gdp"])[current_time]
        return (-1 * (current_gdp - value) ** 2) / ((950 * self.W / 500) ** 2)

        

