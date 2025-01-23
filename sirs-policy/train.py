from sirs_gov import SIRSGov
from model import VanillaModelManager, RARLModelManager, ModelTester, ARLModelManager
from copy import deepcopy
import os
import json
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecNormalize
import time




class TrainPolicies:
    """
    Takes in a particular "vanilla" configuration and a list of adversarial training parameters and trains ARL and RARL policies based on that

    args:
        env_config: base config -- should contain at least observation_types, action_types, reward_type, and additional["alpha"] if required
        train_params: N_iter, N_mu, any other model configs
        dir: the location where to store the models
            assumption: the dir will store 1 vanilla, 1 arl, and 1 rarl model -- 1 dir will be for 1 kind of adversarial training
    """
    def __init__(
        self,
        env_config = {},
        train_params = {},
        adv_action_types: list = ["beta"],
        gym_spaces_bounds: dict = {"act_beta" : (-5.0, 1.0)},
        n_seeds: int = 3,
        dir: str = None
    ):
        if dir == None:
            raise ValueError("dir cannot be None") 
        self.dir = dir

        self.env_config = env_config

        self.n_seeds = n_seeds
        self.seeds = np.array([i for i in range(42, 42 + self.n_seeds)], dtype = np.int32)


        if "N_iter" not in train_params:
            train_params["N_iter"] = 300
        if "N_mu" not in train_params:
            train_params["N_mu"] = 1
        self.train_params = train_params

        self._train_vanilla()

        # print(f"\nprinting gsb inside TrainPolicies")
        # print(gym_spaces_bounds)

        self.sim_config = self.vanilla_model.train_env.get_adv_config(
            adv_action_types = adv_action_types,
            gym_spaces_bounds = gym_spaces_bounds
        )

        # print(self.sim_config)

        self._train_arl()

        self._train_rarl()


    def _train_vanilla(
        self
    ):
        for seed in self.seeds:
            this_dir = os.path.join(self.dir, str(seed))
            os.makedirs(this_dir, exist_ok = True)

            env_config = deepcopy(self.env_config)
            env_config["seed"] = seed

            self.vanilla_model = VanillaModelManager(
                env_config = env_config,
                dir = this_dir,
                seed = seed
                # additional = self.model_additional
            )

            self.vanilla_model.make()

            self.vanilla_model.train_stop(
                N_iter = self.train_params["N_iter"],
                N_mu = self.train_params["N_mu"]
            )

    def _train_arl(
        self
    ):
        for seed in self.seeds:
            this_dir = os.path.join(self.dir, str(seed))
            os.makedirs(this_dir, exist_ok = True)

            sim_config = deepcopy(self.sim_config)
            # sim_config["seed"] = seed

            self.arl_model = ARLModelManager(
                sim_config = sim_config,
                dir = this_dir,
                # additional = self.model_additional,
                seed = seed
            )

            self.arl_model.make()

            self.arl_model.train_stop(
                N_iter = self.train_params["N_iter"],
                N_mu = self.train_params["N_mu"]
            )

    def _train_rarl(
        self
    ):
        for seed in self.seeds:
            this_dir = os.path.join(self.dir, str(seed))
            os.makedirs(this_dir, exist_ok = True)

            sim_config = deepcopy(self.sim_config)
            # sim_config["seed"] = seed

            self.rarl_model = RARLModelManager(
                sim_config = sim_config,
                dir = this_dir,
                # additional = self.model_additional,
                seed = seed
            )

            self.rarl_model.make()

            self.rarl_model.train_stop(
                N_iter = self.train_params["N_iter"],
                N_mu = self.train_params["N_mu"]
            )



# def dict_to_string(self, input_dict):
#     result = []
#     for key, value in input_dict.items():
#         abbrev = get_abbreviation(key)
#         value_str = '_'.join(map(str, value))
#         result.append((abbrev, f"{abbrev}_{value_str}"))
#     # Sort the result list by abbreviation
#     result.sort(key=lambda x: x[0])
#     # Concatenate the second elements (abbrev + values) with '_'
#     output_string = '_'.join(item[1] for item in result)
#     return output_string

# def get_abbreviation(self, key):
#     words = key.split('_')
#     if words[0] == 'act' or words[0] == 'obs':
#         words = words[1:]
#     abbrev = ''.join(word[0].upper() for word in words)
#     return abbrev


