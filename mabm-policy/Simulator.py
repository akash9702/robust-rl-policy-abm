import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import random

from typing import List, Any, Dict, Tuple, Callable

import juliacall
from juliacall import Main as jl

import matplotlib.pyplot as plt
from copy import deepcopy

from gymnasium.utils import seeding

class ARLABCSimulator():
    _default_gym_spaces_bounds: Dict = {
        "act_income_tax": (0.0, 1.0),
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
        "act_subsidy": (0.0, 1.0),
        "act_consumption_wealth_ratio": (0.0, 1.0),
        "act_firm_invest_prob": (0.0, 1.0),
        "act_memory_parameter": (0.0, 1.0),
        "act_labour_prod" : (0.0, 1.0),
        "act_capital_prod" : (0.0, 1.0)
    }

    def __init__(
        self,
        T: int = 300,
        W: int = 250,
        F: int = 25,
        N: int = 5,
        t_burnin: int = 30,
        n_const_steps: int = 1,
        n_prev_obs: int = 2,
        gym_spaces_bounds: Dict[str, Tuple[float, float]] = {},
        gov_action_types: List[str] = ["income_tax", "subsidy"],
        adv_action_types: List[str] = ["consumption_wealth_ratio"],
        reward_type: str = "gdp_growth_stability",
        gov_observation_types: List[str] = ["gdp", "bank_deposits"],
        adv_observation_types: List[str] = ["gdp", "bank_deposits"],
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

        self.gov_allowed_action_types = [
            "income_tax",
            "subsidy",
            "interest_rate"
        ]

        self.adv_allowed_action_types = [
            "firm_invest_prob",
            "consumption_wealth_ratio",
            "memory_parameter",
            "labour_prod",
            "capital_prod"
        ]

        self.actions_to_properties = {
            "income_tax" : "tax_rate",
            "firm_invest_prob" : "Iprob",
            "consumption_wealth_ratio" : "chi",
            "subsidy": "subsidy",
            "interest_rate": "interest_rate",
            "memory_parameter": "xi",
            "labour_prod" : "alpha",
            "capital_prod" : "k"
        }

        self.gov_allowed_observation_types = [
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

        self.adv_allowed_observation_types = [
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
            "gdp_growth_stability"
        ]

        # dictionary storing different reward functions 
        self.reward_functions = {
            "gdp_growth_absolute" : self._gdp_growth_absolute_reward,
            "gdp_growth_percentage" : self._gdp_growth_percentage_reward,
            "gdp_growth_deficit" :  self._gdp_growth_deficit_reward,
            "gdp_growth_stability" : self._gdp_growth_stability_reward
        }

        if not self._check_proper_reward(reward_type):
            raise ValueError("reward should be one of " + str(self.allowed_reward_types))
        if not self._check_proper_params(params):
            raise ValueError("params should be one of " + str(self.allowed_params))
        self._check_proper_actions(gov_action_types, adv_action_types)
        self._check_proper_observations(gov_observation_types, adv_observation_types)
            
        # seeding gym
        # seeding.np_random(self.seed)
        
        # import ABCredit and initialise the model through Julia
        # jl.seval("using Pkg")
        # jl.seval("Pkg.add(\"ABCredit\")")
        # jl.seval("Pkg.instantiate()")
        jl.seval("using ABCredit")
        jl.seval("using Random")
        # setting random seed to self.seed
        jl.seval("Random.seed!")(self.seed)
        self._julia_model_init()

        
        self._create_gym_spaces_bounds(gym_spaces_bounds)

        self._create_spaces_gov()
        self._create_spaces_adv()

    def step(
        self,
        action
    ):
        self.step_number += 1

        # if self.step_number % 50 == 0: 
        #     print(f"step {self.step_number}")

        action_np = np.array(action, dtype = np.float32)

        for i in range(action_np.shape[0]):
            self.model.params[jl.Symbol(self.actions_to_properties[self.gov_action_types[i]])] = action_np[i]

        for i in range(self.n_const_steps):
            self.t += 1
            # print(f"simulation timestep {self.model.agg.timestep}")
            jl.seval("one_model_step!")(self.model)
            jl.seval("ABCredit.update_data!")(self.data, self.model)

            if self.t >= self.T:
                break
        
        reward = self._get_reward()

        obs = self._get_obs_gov()
        

        info = self._get_info()
        terminated = self._get_terminated()
        truncated = self._get_truncated()

        # self.state = obs
        
        return obs, reward, terminated, truncated, info

    def reset_simulator(
        self,
        train_gov: bool
    ):
  
        self.t = 0
        self.step_number = 0

        self._julia_model_init()

        if train_gov:
            obs = self._get_obs_gov()
        else:
            obs = self._get_obs_adv()
        
        # print(f"episode {self.episode_number}")

        return obs, self._get_info()

    def render(
        self
    ):
        pass

    def get_config(
        self
    ):
        config = deepcopy(vars(self))

        keys_to_remove = [
            "gov_allowed_action_types", 
            "adv_allowed_action_types", 
            "gov_allowed_observation_types", 
            "adv_allowed_observation_types", 
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
            "gov_action_space", 
            "adv_action_space", 
            "gov_observation_space",
            "adv_observation_space"
        ]

        for key in keys_to_remove:
            if key in config:
                del config[key]

        return config
    
    def take_adv_action(
        self,
        action
    ):
        action_np = np.array(action, dtype = np.float32)

        for i in range(action_np.shape[0]):
            self.model.params[jl.Symbol(self.actions_to_properties[self.adv_action_types[i]])] = action_np[i]

    def _get_reward(
        self
    ):
        return self.reward_functions[self.reward_type]()

    def _get_terminated(
        self
    ):
        terminated = self.t >= self.T

        if terminated:
            # print("env to be terminated")
            
            if self.episode_number % self.plot_after_every == 0 and self.plot:
                self._plot_env()
            
            # print(f"episode {self.episode_number} terminated")
            
            self.episode_number += 1
            

        return terminated

    def _get_truncated(
        self
    ):
        return self.t >= self.T
    
    def get_obs_sim(
        self,
        gov
    ):
        if gov:
            return self._get_obs_gov()
        else:
            return self._get_obs_adv()
        
    def get_info_sim(
        self
    ):
        return self._get_info()
    
    def _get_obs_gov(
        self
    ):
        return self._get_obs_helper("gov")
    
    def _get_obs_adv(
        self
    ):
        return self._get_obs_helper("adv")
    
    def _get_obs_helper(
        self,
        which: str = "gov"
    ):
        obs_type_array = None
        if which == "gov":
            obs_type_array = self.gov_observation_types
        else:
            obs_type_array = self.adv_observation_types

        current_time = self.model.agg.timestep - 1

        n_observation_types = len(obs_type_array)
        obs = np.zeros(shape = (n_observation_types, self.n_prev_obs), dtype = np.float32)

        n_real_observations = min(self.n_prev_obs, current_time)
        obs_no = 0
        for observation in obs_type_array:
            for iter in range(n_real_observations):
                attribute_value = getattr(self.data, self.observations_to_properties[observation])[current_time - n_real_observations + iter + 1]
                obs[obs_no][self.n_prev_obs - n_real_observations + iter] = attribute_value

            # if some observations are still 0 then set them to the value of the last known observation
            if n_real_observations > 0:
                obs[obs_no][0:self.n_prev_obs - n_real_observations] = obs[obs_no][self.n_prev_obs - n_real_observations]


            # for iter in range(self.n_prev_obs - n_real_observations):
            #     obs[obs_no][iter] = obs[obs_no][self.n_prev_obs - n_real_observations]
            obs_no += 1

        return obs
        
        # current_time = self.model.agg.timestep - 1

        # n_observations = len(obs_type_array)
        # obs = np.zeros(shape = (n_observations, ), dtype = np.float32)

        # obs_no = 0
        # for observation in obs_type_array:
        #     attribute_value = getattr(self.data, self.observations_to_properties[observation])[current_time]
        #     obs[obs_no] = attribute_value
        #     obs_no += 1

        # return obs
    
    def get_info_sim(
        self
    ):
        return self._get_info()

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

        for action in self.gov_action_types:
            action_key = str("act_" + action)
            if action_key not in gym_spaces_bounds:
                self.gym_spaces_bounds[action_key] = self._default_gym_spaces_bounds[action_key]
            else:
                self.gym_spaces_bounds[action_key] = gym_spaces_bounds[action_key]
        
        for observation in self.gov_observation_types:
            observation_key = str("obs_" + observation)
            if observation_key not in gym_spaces_bounds:
                self.gym_spaces_bounds[observation_key] = self._default_gym_spaces_bounds[observation_key]
            else:
                self.gym_spaces_bounds[observation_key] = gym_spaces_bounds[observation_key]

        for action in self.adv_action_types:
            action_key = str("act_" + action)
            if action_key not in gym_spaces_bounds:
                self.gym_spaces_bounds[action_key] = self._default_gym_spaces_bounds[action_key]
            else:
                self.gym_spaces_bounds[action_key] = gym_spaces_bounds[action_key]
        
        for observation in self.adv_observation_types:
            observation_key = str("obs_" + observation)
            if observation_key not in gym_spaces_bounds:
                self.gym_spaces_bounds[observation_key] = self._default_gym_spaces_bounds[observation_key]
            else:
                self.gym_spaces_bounds[observation_key] = gym_spaces_bounds[observation_key]

        # for action in self.gov_action_types:
        #     action_key = str("act_" + action)
        #     self.gym_spaces_bounds[action_key] = self._default_gym_spaces_bounds[action_key]

        # for action in self.adv_action_types:
        #     action_key = str("act_" + action)
        #     self.gym_spaces_bounds[action_key] = self._default_gym_spaces_bounds[action_key]
        
        # for observation in self.gov_observation_types:
        #     observation_key = str("obs_" + observation)
        #     self.gym_spaces_bounds[observation_key] = self._default_gym_spaces_bounds[observation_key]

        # for observation in self.adv_observation_types:
        #     observation_key = str("obs_" + observation)
        #     self.gym_spaces_bounds[observation_key] = self._default_gym_spaces_bounds[observation_key]
        
    # TODO: implement this
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

    def _julia_model_init(
        self
    ):
        # making (for instance) ABCredit.PARAMS_ORIGINAL as a Julia object
        params_jl = deepcopy(jl.seval(str("ABCredit." + self.params)))

        # change behavioural parameters before model creation
        if "change" in self.additional and self.additional["change"]:
            for param, new_value in self.additional["params_to_change"].items():
                params_jl[jl.Symbol(self.actions_to_properties[param])] = new_value

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
            gov_action_types,
            adv_action_types
    ) -> bool:
        """
        checks that the gov_action_types and adv_action_types given by the user are proper actions 
        """

        if set(gov_action_types).issubset(set(self.gov_allowed_action_types)):
            self.gov_action_types = gov_action_types
        else:
            raise ValueError("gov_action_types should be subset of " + str(self.gov_allowed_action_types))
        
        if set(adv_action_types).issubset(set(self.adv_allowed_action_types)):
            self.adv_action_types = adv_action_types
        else:
            raise ValueError("adv_action_types should be subset of " + str(self.adv_allowed_action_types))
        
            
    def _check_proper_observations(
        self,
        gov_observation_types,
        adv_observation_types
    ):
        """
        checks that the observation_types given by the user are proper observations
        """

        if set(gov_observation_types).issubset(set(self.gov_allowed_observation_types)):
            self.gov_observation_types = gov_observation_types
        else:
            raise ValueError("gov_observation_types should be subset of " + str(self.gov_observation_types))
        
        if set(adv_observation_types).issubset(set(self.adv_allowed_observation_types)):
            self.adv_observation_types = adv_observation_types
        else:
            raise ValueError("adv_observation_types should be subset of " + str(self.adv_allowed_observation_types))
            
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
        
    def _create_spaces_gov(
            self
    ) -> None:
        """
        creates the gov_action_space and the gov_observation_space 
        """

        number_of_actions = int(len(self.gov_action_types))

        # num_actions_in_bounds_dict = 0
        # for variable, space_range in self.gym_spaces_bounds.items():
        #     if variable[: 3] == "act":
        #         num_actions_in_bounds_dict += 1
        
        # if not num_actions_in_bounds_dict == number_of_actions:
        #     raise ValueError("Number of action spaces in gym spaces bounds doesn't match number of action types.")

        actions_low_array = np.zeros(shape = (number_of_actions, ), dtype = np.float32)
        actions_high_array = np.zeros(shape = (number_of_actions, ), dtype = np.float32)

        # TODO: put a check on the face that the names in gym_spaces_bounds should match action and observation names
        for i in range(number_of_actions):
            action = self.gov_action_types[i]
            actions_low_array[i] = self.gym_spaces_bounds[str("act_" + action)][0]
            actions_high_array[i] = self.gym_spaces_bounds[str("act_" + action)][1]
        
        self.gov_action_space = spaces.Box(
            low = actions_low_array,
            high = actions_high_array,
            shape = (number_of_actions, ),
            dtype = np.float32
        )

        n_observation_types = len(self.gov_observation_types)
        observations_low_array = np.zeros(shape = (n_observation_types, self.n_prev_obs), dtype = np.float32)
        observations_high_array = np.zeros(shape = (n_observation_types, self.n_prev_obs), dtype = np.float32)

        for i in range(n_observation_types):
            observation = self.gov_observation_types[i]
            for j in range(self.n_prev_obs):
                observations_low_array[i][j] = self.gym_spaces_bounds[str("obs_" + observation)][0]
                observations_high_array[i][j] = self.gym_spaces_bounds[str("obs_" + observation)][1]

        self.gov_observation_space = spaces.Box(
            low = observations_low_array,
            high = observations_high_array,
            shape = (n_observation_types, self.n_prev_obs),
            dtype = np.float32
        )

        # number_of_observations = len(self.gov_observation_types)

        # observations_low_array = np.zeros(shape = (number_of_observations, ), dtype = np.float32)
        # observations_high_array = np.zeros(shape = (number_of_observations, ), dtype = np.float32)
        

        # for i in range(number_of_observations):
        #     observation = self.gov_observation_types[i]
        #     observations_low_array[i] = self.gym_spaces_bounds[str("obs_" + observation)][0]
        #     observations_high_array[i] = self.gym_spaces_bounds[str("obs_" + observation)][1]

        # self.gov_observation_space = spaces.Box(
        #     low = observations_low_array,
        #     high = observations_high_array,
        #     shape = (number_of_observations, ),
        #     dtype = np.float32
        # )

    def _create_spaces_adv(
            self
    ) -> None:
        """
        creates the adv_action_space and the adv_observation_space 
        """

        number_of_actions = int(len(self.adv_action_types))

        # num_actions_in_bounds_dict = 0
        # for variable, space_range in self.gym_spaces_bounds.items():
        #     if variable[: 3] == "act":
        #         num_actions_in_bounds_dict += 1
        
        # if not num_actions_in_bounds_dict == number_of_actions:
        #     raise ValueError("Number of action spaces in gym spaces bounds doesn't match number of action types.")

        actions_low_array = np.zeros(shape = (number_of_actions, ), dtype = np.float32)
        actions_high_array = np.zeros(shape = (number_of_actions, ), dtype = np.float32)

        # TODO: put a check on the face that the names in gym_spaces_bounds should match action and observation names
        for i in range(number_of_actions):
            action = self.adv_action_types[i]
            actions_low_array[i] = self.gym_spaces_bounds[str("act_" + action)][0]
            actions_high_array[i] = self.gym_spaces_bounds[str("act_" + action)][1]
        
        self.adv_action_space = spaces.Box(
            low = actions_low_array,
            high = actions_high_array,
            shape = (number_of_actions, ),
            dtype = np.float32
        )

        n_observation_types = len(self.adv_observation_types)
        observations_low_array = np.zeros(shape = (n_observation_types, self.n_prev_obs), dtype = np.float32)
        observations_high_array = np.zeros(shape = (n_observation_types, self.n_prev_obs), dtype = np.float32)

        for i in range(n_observation_types):
            observation = self.adv_observation_types[i]
            for j in range(self.n_prev_obs):
                observations_low_array[i][j] = self.gym_spaces_bounds[str("obs_" + observation)][0]
                observations_high_array[i][j] = self.gym_spaces_bounds[str("obs_" + observation)][1]

        self.adv_observation_space = spaces.Box(
            low = observations_low_array,
            high = observations_high_array,
            shape = (n_observation_types, self.n_prev_obs),
            dtype = np.float32
        )

        # number_of_observations = len(self.adv_observation_types)

        # observations_low_array = np.zeros(shape = (number_of_observations, ), dtype = np.float32)
        # observations_high_array = np.zeros(shape = (number_of_observations, ), dtype = np.float32)
        

        # for i in range(number_of_observations):
        #     observation = self.adv_observation_types[i]
        #     observations_low_array[i] = self.gym_spaces_bounds[str("obs_" + observation)][0]
        #     observations_high_array[i] = self.gym_spaces_bounds[str("obs_" + observation)][1]

        # self.adv_observation_space = spaces.Box(
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

        return (current_gdp - last_gdp) - alpha * (current_deficit - last_deficit)

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

    

class AdvGovABCSimulator(gym.Env):
    """
    Implementation of the Model Simulator class for the adversarial training of a single government agent on the ABCredit simulator.

    This isn't a typical gym environment and is not meant to be made compatible with gym related libraries. In fact, it shouldn't even necessarily inherit from gym.Env (we can change this and it won't affect anything). There are two environments -- GovABC and AdvABC -- that use this simulator to find rewards etc of their actions.

    All of the details relevant only to the simulator are implemented here. Further, some other details (like whether the gov_action_types are in gov_allowed_action_types) are also implemented here since we will be making objects of this class first and so checking everything here makes more sense (so that we don't even proceed to making the gym envs until the values are correct).

    Args:
        T: Number of steps in the simulation
        W: Number of workers
        F: Number of consumption-goods producing firms
        N: Number of capital-goods producing firms
        params: Parameters to be used by the model
        reward_type: the reward type for the government. 

    Args to be implemented:
        info_level: Level of information returned by the environment after `step` is called. 0 returns no information
            (empty dictionary), 1 returns information used for plotting, 2 returns all information (default).
        
    """

    _default_gym_spaces_bounds: Dict = {
        "act_income_tax": (0.0, 1.0),
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
        "act_subsidy": (0.0, 1.0),
        "act_consumption_wealth_ratio": (0.0, 1.0),
        "act_firm_invest_prob": (0.0, 1.0),
        "act_memory_parameter": (0.0, 1.0),
        "act_labour_prod" : (0.0, 1.0),
        "act_capital_prod" : (0.0, 1.0)
    }

    # _default_gym_spaces_bounds: Dict = {
    #     "act_income_tax": (0.0, 1.0),
    #     "obs_gdp": (0, 1e20),
    #     "obs_gdp_deficit": (-1e60, 1e60),
    #     "act_firm_invest_prob": (0.1, 0.5),
    #     "act_consumption_wealth_ratio": (0.01, 0.1),
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
    #     "act_subsidy": (0.0, 1.0),
    #     "act_memory_parameter": (0.0, 1.0)
    # }

    def __init__(
        self,
        T: int = 300,
        W: int = 250,
        F: int = 25,
        N: int = 5,
        t_burnin: int = 30,
        n_const_steps: int = 1,
        n_prev_obs: int = 2,
        gym_spaces_bounds: Dict[str, Tuple[float, float]] = {},
        gov_action_types: List[str] = ["income_tax", "subsidy"],
        adv_action_types: List[str] = ["consumption_wealth_ratio"],
        reward_type: str = "gdp_growth_stability",
        gov_observation_types: List[str] = ["gdp", "bank_deposits"],
        adv_observation_types: List[str] = ["gdp", "bank_deposits"],
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

        self.gov_allowed_action_types = [
            "income_tax",
            "subsidy",
            "interest_rate"
        ]

        self.adv_allowed_action_types = [
            "firm_invest_prob",
            "consumption_wealth_ratio",
            "memory_parameter",
            "labour_prod",
            "capital_prod"
        ]

        self.actions_to_properties = {
            "income_tax" : "tax_rate",
            "firm_invest_prob" : "Iprob",
            "consumption_wealth_ratio" : "chi",
            "subsidy": "subsidy",
            "interest_rate": "interest_rate",
            "memory_parameter": "xi",
            "labour_prod" : "alpha",
            "capital_prod" : "k"
        }

        self.gov_allowed_observation_types = [
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

        self.adv_allowed_observation_types = [
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
            "gdp_growth_stability"
        ]

        # dictionary storing different reward functions 
        self.reward_functions = {
            "gdp_growth_absolute" : self._gdp_growth_absolute_reward,
            "gdp_growth_percentage" : self._gdp_growth_percentage_reward,
            "gdp_growth_deficit" :  self._gdp_growth_deficit_reward,
            "gdp_growth_stability" : self._gdp_growth_stability_reward
        }

        if not self._check_proper_reward(reward_type):
            raise ValueError("reward should be one of " + str(self.allowed_reward_types))
        if not self._check_proper_params(params):
            raise ValueError("params should be one of " + str(self.allowed_params))
        self._check_proper_actions(gov_action_types, adv_action_types)
        self._check_proper_observations(gov_observation_types, adv_observation_types)
            
        # seeding gym
        # seeding.np_random(self.seed)
        
        # import ABCredit and initialise the model through Julia
        # jl.seval("using Pkg")
        # jl.seval("Pkg.add(\"ABCredit\")")
        # jl.seval("Pkg.instantiate()")
        jl.seval("using ABCredit")
        jl.seval("using Random")
        # setting random seed to self.seed
        jl.seval("Random.seed!")(self.seed)
        self._julia_model_init()

        
        self._create_gym_spaces_bounds(gym_spaces_bounds)

        self._create_spaces_gov()
        self._create_spaces_adv()

        # self.state = self._get_obs().item()

    def step(
        self,
        gov_action,
        adv_action,
        train_gov: bool
    ):
        self.step_number += 1

        # print(f"step {self.step_number}")

        gov_action_np = np.array(gov_action, dtype = np.float32)
        adv_action_np = np.array(adv_action, dtype = np.float32)

        for i in range(gov_action_np.shape[0]):
            self.model.params[jl.Symbol(self.actions_to_properties[self.gov_action_types[i]])] = gov_action_np[i]

        for i in range(adv_action_np.shape[0]):
            self.model.params[jl.Symbol(self.actions_to_properties[self.adv_action_types[i]])] = adv_action_np[i]

        for i in range(self.n_const_steps):
            self.t += 1
            # print(f"simulation timestep {self.model.agg.timestep}")
            jl.seval("one_model_step!")(self.model)
            jl.seval("ABCredit.update_data!")(self.data, self.model)

            if self.t >= self.T:
                break
        
        reward = self._get_reward()

        if train_gov:
            obs = self._get_obs_gov()
        else:
            obs = self._get_obs_adv()
            reward = -reward

        info = self._get_info()
        terminated = self._get_terminated()
        truncated = self._get_truncated()

        # self.state = obs
        
        return obs, reward, terminated, truncated, info

    def reset_simulator(
        self,
        train_gov: bool
    ):
  
        self.t = 0
        self.step_number = 0

        self._julia_model_init()

        if train_gov:
            obs = self._get_obs_gov()
        else:
            obs = self._get_obs_adv()

        return obs, self._get_info()

    def render(
        self
    ):
        pass

    def get_config(
        self
    ):
        config = deepcopy(vars(self))

        keys_to_remove = [
            "gov_allowed_action_types", 
            "adv_allowed_action_types", 
            "gov_allowed_observation_types", 
            "adv_allowed_observation_types", 
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
            "gov_action_space", 
            "adv_action_space", 
            "gov_observation_space",
            "adv_observation_space"
        ]

        for key in keys_to_remove:
            if key in config:
                del config[key]

        return config

    def _get_reward(
        self
    ):
        return self.reward_functions[self.reward_type]()

    def _get_terminated(
        self
    ):
        terminated = self.t >= self.T

        if terminated:
            # print("env to be terminated")
            
            if self.episode_number % self.plot_after_every == 0 and self.plot:
                self._plot_env()
            
            
            self.episode_number += 1

        return terminated

    def _get_truncated(
        self
    ):
        return self.t >= self.T
    
    def get_obs_sim(
        self,
        gov
    ):
        if gov:
            return self._get_obs_gov()
        else:
            return self._get_obs_adv()
    
    def _get_obs_gov(
        self
    ):
        return self._get_obs_helper("gov")
    
    def _get_obs_adv(
        self
    ):
        return self._get_obs_helper("adv")
    
    def _get_obs_helper(
        self,
        which: str = "gov"
    ):
        obs_type_array = None
        if which == "gov":
            obs_type_array = self.gov_observation_types
        else:
            obs_type_array = self.adv_observation_types

        current_time = self.model.agg.timestep - 1

        n_observation_types = len(obs_type_array)
        obs = np.zeros(shape = (n_observation_types, self.n_prev_obs), dtype = np.float32)

        n_real_observations = min(self.n_prev_obs, current_time)
        obs_no = 0
        for observation in obs_type_array:
            for iter in range(n_real_observations):
                attribute_value = getattr(self.data, self.observations_to_properties[observation])[current_time - n_real_observations + iter + 1]
                obs[obs_no][self.n_prev_obs - n_real_observations + iter] = attribute_value

            # if some observations are still 0 then set them to the value of the last known observation
            if n_real_observations > 0:
                obs[obs_no][0:self.n_prev_obs - n_real_observations] = obs[obs_no][self.n_prev_obs - n_real_observations]


            # for iter in range(self.n_prev_obs - n_real_observations):
            #     obs[obs_no][iter] = obs[obs_no][self.n_prev_obs - n_real_observations]
            obs_no += 1

        return obs
        
        # current_time = self.model.agg.timestep - 1

        # n_observations = len(obs_type_array)
        # obs = np.zeros(shape = (n_observations, ), dtype = np.float32)

        # obs_no = 0
        # for observation in obs_type_array:
        #     attribute_value = getattr(self.data, self.observations_to_properties[observation])[current_time]
        #     obs[obs_no] = attribute_value
        #     obs_no += 1

        # return obs
    
    def get_info_sim(
        self
    ):
        return self._get_info()

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

        for action in self.gov_action_types:
            action_key = str("act_" + action)
            if action_key not in gym_spaces_bounds:
                self.gym_spaces_bounds[action_key] = self._default_gym_spaces_bounds[action_key]
            else:
                self.gym_spaces_bounds[action_key] = gym_spaces_bounds[action_key]
        
        for observation in self.gov_observation_types:
            observation_key = str("obs_" + observation)
            if observation_key not in gym_spaces_bounds:
                self.gym_spaces_bounds[observation_key] = self._default_gym_spaces_bounds[observation_key]
            else:
                self.gym_spaces_bounds[observation_key] = gym_spaces_bounds[observation_key]

        for action in self.adv_action_types:
            action_key = str("act_" + action)
            if action_key not in gym_spaces_bounds:
                self.gym_spaces_bounds[action_key] = self._default_gym_spaces_bounds[action_key]
            else:
                self.gym_spaces_bounds[action_key] = gym_spaces_bounds[action_key]
        
        for observation in self.adv_observation_types:
            observation_key = str("obs_" + observation)
            if observation_key not in gym_spaces_bounds:
                self.gym_spaces_bounds[observation_key] = self._default_gym_spaces_bounds[observation_key]
            else:
                self.gym_spaces_bounds[observation_key] = gym_spaces_bounds[observation_key]

        # for action in self.gov_action_types:
        #     action_key = str("act_" + action)
        #     self.gym_spaces_bounds[action_key] = self._default_gym_spaces_bounds[action_key]

        # for action in self.adv_action_types:
        #     action_key = str("act_" + action)
        #     self.gym_spaces_bounds[action_key] = self._default_gym_spaces_bounds[action_key]
        
        # for observation in self.gov_observation_types:
        #     observation_key = str("obs_" + observation)
        #     self.gym_spaces_bounds[observation_key] = self._default_gym_spaces_bounds[observation_key]

        # for observation in self.adv_observation_types:
        #     observation_key = str("obs_" + observation)
        #     self.gym_spaces_bounds[observation_key] = self._default_gym_spaces_bounds[observation_key]
        
    # TODO: implement this
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

    def _julia_model_init(
        self
    ):
        # making (for instance) ABCredit.PARAMS_ORIGINAL as a Julia object
        params_jl = deepcopy(jl.seval(str("ABCredit." + self.params)))

        # setting random seed to nothing (since that's the default in the ABCredit code)
        # # jl.seval("Random.seed!(nothing)")
        # jl.seval("Random.seed!")(self.seed)

        # change behavioural parameters before model creation
        if "change" in self.additional and self.additional["change"]:
            for param, new_value in self.additional["params_to_change"].items():
                params_jl[jl.Symbol(self.actions_to_properties[param])] = new_value

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
            gov_action_types,
            adv_action_types
    ) -> bool:
        """
        checks that the gov_action_types and adv_action_types given by the user are proper actions 
        """

        if set(gov_action_types).issubset(set(self.gov_allowed_action_types)):
            self.gov_action_types = gov_action_types
        else:
            raise ValueError("gov_action_types should be subset of " + str(self.gov_allowed_action_types))
        
        if set(adv_action_types).issubset(set(self.adv_allowed_action_types)):
            self.adv_action_types = adv_action_types
        else:
            raise ValueError("adv_action_types should be subset of " + str(self.adv_allowed_action_types))
        
            
    def _check_proper_observations(
        self,
        gov_observation_types,
        adv_observation_types
    ):
        """
        checks that the observation_types given by the user are proper observations
        """

        if set(gov_observation_types).issubset(set(self.gov_allowed_observation_types)):
            self.gov_observation_types = gov_observation_types
        else:
            raise ValueError("gov_observation_types should be subset of " + str(self.gov_observation_types))
        
        if set(adv_observation_types).issubset(set(self.adv_allowed_observation_types)):
            self.adv_observation_types = adv_observation_types
        else:
            raise ValueError("adv_observation_types should be subset of " + str(self.adv_allowed_observation_types))
            
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
        
    def _create_spaces_gov(
            self
    ) -> None:
        """
        creates the gov_action_space and the gov_observation_space 
        """

        number_of_actions = int(len(self.gov_action_types))

        # num_actions_in_bounds_dict = 0
        # for variable, space_range in self.gym_spaces_bounds.items():
        #     if variable[: 3] == "act":
        #         num_actions_in_bounds_dict += 1
        
        # if not num_actions_in_bounds_dict == number_of_actions:
        #     raise ValueError("Number of action spaces in gym spaces bounds doesn't match number of action types.")

        actions_low_array = np.zeros(shape = (number_of_actions, ), dtype = np.float32)
        actions_high_array = np.zeros(shape = (number_of_actions, ), dtype = np.float32)

        # TODO: put a check on the face that the names in gym_spaces_bounds should match action and observation names
        for i in range(number_of_actions):
            action = self.gov_action_types[i]
            actions_low_array[i] = self.gym_spaces_bounds[str("act_" + action)][0]
            actions_high_array[i] = self.gym_spaces_bounds[str("act_" + action)][1]
        
        self.gov_action_space = spaces.Box(
            low = actions_low_array,
            high = actions_high_array,
            shape = (number_of_actions, ),
            dtype = np.float32
        )

        n_observation_types = len(self.gov_observation_types)
        observations_low_array = np.zeros(shape = (n_observation_types, self.n_prev_obs), dtype = np.float32)
        observations_high_array = np.zeros(shape = (n_observation_types, self.n_prev_obs), dtype = np.float32)

        for i in range(n_observation_types):
            observation = self.gov_observation_types[i]
            for j in range(self.n_prev_obs):
                observations_low_array[i][j] = self.gym_spaces_bounds[str("obs_" + observation)][0]
                observations_high_array[i][j] = self.gym_spaces_bounds[str("obs_" + observation)][1]

        self.gov_observation_space = spaces.Box(
            low = observations_low_array,
            high = observations_high_array,
            shape = (n_observation_types, self.n_prev_obs),
            dtype = np.float32
        )

        # number_of_observations = len(self.gov_observation_types)

        # observations_low_array = np.zeros(shape = (number_of_observations, ), dtype = np.float32)
        # observations_high_array = np.zeros(shape = (number_of_observations, ), dtype = np.float32)
        

        # for i in range(number_of_observations):
        #     observation = self.gov_observation_types[i]
        #     observations_low_array[i] = self.gym_spaces_bounds[str("obs_" + observation)][0]
        #     observations_high_array[i] = self.gym_spaces_bounds[str("obs_" + observation)][1]

        # self.gov_observation_space = spaces.Box(
        #     low = observations_low_array,
        #     high = observations_high_array,
        #     shape = (number_of_observations, ),
        #     dtype = np.float32
        # )

    def _create_spaces_adv(
            self
    ) -> None:
        """
        creates the adv_action_space and the adv_observation_space 
        """

        number_of_actions = int(len(self.adv_action_types))

        # num_actions_in_bounds_dict = 0
        # for variable, space_range in self.gym_spaces_bounds.items():
        #     if variable[: 3] == "act":
        #         num_actions_in_bounds_dict += 1
        
        # if not num_actions_in_bounds_dict == number_of_actions:
        #     raise ValueError("Number of action spaces in gym spaces bounds doesn't match number of action types.")

        actions_low_array = np.zeros(shape = (number_of_actions, ), dtype = np.float32)
        actions_high_array = np.zeros(shape = (number_of_actions, ), dtype = np.float32)

        # TODO: put a check on the face that the names in gym_spaces_bounds should match action and observation names
        for i in range(number_of_actions):
            action = self.adv_action_types[i]
            actions_low_array[i] = self.gym_spaces_bounds[str("act_" + action)][0]
            actions_high_array[i] = self.gym_spaces_bounds[str("act_" + action)][1]
        
        self.adv_action_space = spaces.Box(
            low = actions_low_array,
            high = actions_high_array,
            shape = (number_of_actions, ),
            dtype = np.float32
        )

        n_observation_types = len(self.adv_observation_types)
        observations_low_array = np.zeros(shape = (n_observation_types, self.n_prev_obs), dtype = np.float32)
        observations_high_array = np.zeros(shape = (n_observation_types, self.n_prev_obs), dtype = np.float32)

        for i in range(n_observation_types):
            observation = self.adv_observation_types[i]
            for j in range(self.n_prev_obs):
                observations_low_array[i][j] = self.gym_spaces_bounds[str("obs_" + observation)][0]
                observations_high_array[i][j] = self.gym_spaces_bounds[str("obs_" + observation)][1]

        self.adv_observation_space = spaces.Box(
            low = observations_low_array,
            high = observations_high_array,
            shape = (n_observation_types, self.n_prev_obs),
            dtype = np.float32
        )

        # number_of_observations = len(self.adv_observation_types)

        # observations_low_array = np.zeros(shape = (number_of_observations, ), dtype = np.float32)
        # observations_high_array = np.zeros(shape = (number_of_observations, ), dtype = np.float32)
        

        # for i in range(number_of_observations):
        #     observation = self.adv_observation_types[i]
        #     observations_low_array[i] = self.gym_spaces_bounds[str("obs_" + observation)][0]
        #     observations_high_array[i] = self.gym_spaces_bounds[str("obs_" + observation)][1]

        # self.adv_observation_space = spaces.Box(
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

        return (current_gdp - last_gdp) - alpha * (current_deficit - last_deficit)

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