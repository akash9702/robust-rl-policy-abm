from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS

from train import TrainPolicies

import os
import numpy as np

import psutil
import os

def assign_to_least_used_core():
    # Get per-core CPU utilization
    core_loads = psutil.cpu_percent(percpu=True)
    least_used_core = core_loads.index(min(core_loads))  # Find least utilized core
    
    # Set affinity to the least used core
    p = psutil.Process(os.getpid())
    p.cpu_affinity([least_used_core])
    print(f"Assigned to core {least_used_core}")



"""
conda deactivate && conda deactivate && conda activate rlabm
conda deactivate && conda activate rlabm
"""


if __name__ == "__main__":

    assign_to_least_used_core()

    # algo = "SAC"
    # algo = "DDPG"
    # algo = "TD3"
    algo = "PPO"
    # algo = "TRPO"
    # algo = "A2C"

    base_dir = f"../results/test/exps_algos/2_seeds_all_robust/{algo.lower()}"
    base_dir = f"../results/test/feature_selection/{algo.lower()}/both"

    os.makedirs(base_dir, exist_ok = True)

    trainings = {
        "CWR_0.01_0.99" : {
            "adv_action_types" : ["consumption_wealth_ratio"],
            "range" : (0.01, 0.99)
        },
        # "CWR_0.01_0.3" : {
        #     "adv_action_types" : ["consumption_wealth_ratio"],
        #     "range" : (0.01, 0.3)
        # },
        # "FIP_0.1_0.5" : {
        #     "adv_action_types" : ["firm_invest_prob"],
        #     "range" : (0.1, 0.5)
        # },
        # "CP_0.2_0.8" : {
        #     "adv_action_types" : ["capital_prod"],
        #     "range" : (0.2, 0.8)
        # },
        # "LP_0.5_0.75" : {
        #     "adv_action_types" : ["labour_prod"],
        #     "range" : (0.5, 0.75)
        # },
        "MP_0.01_0.99" : {
            "adv_action_types" : ["memory_parameter"],
            "range" : (0.01, 0.99)
        }
    }

    for key in trainings:
        config = trainings[key]

        adv_action_types = config["adv_action_types"]
        adv_range = config["range"]

        gym_spaces_bounds = {
            f"act_{adv_action_types[0]}" : adv_range
        }

        dir = os.path.join(base_dir, key)
        os.makedirs(dir, exist_ok = True)

        N_iter = 500
        if algo.lower() == "ppo":
            N_iter = 4000
        elif algo.lower() == "sac":
            N_iter = 4000
        elif algo.lower() == "ddpg":
            N_iter = 4000
        elif algo.lower() == "td3":
            N_iter = 4000
        N_iter = 1000

        N_iter = 500    # for ppo

        observation_types = [
            "gdp",
            # "gdp_deficit",
            # "consumption",
            # "gov_bonds",
            # "bank_reserves",
            "bank_deposits",
            # "investment",
            # "bank_profits",
            # "bank_dividends",
            # "bank_equity",
            # "gov_balance",
            # "unemployment_rate",
            # "inflation_rate",
            # "bankruptcy_rate"
        ]


        training = TrainPolicies(
            env_config = {
                "action_types" : ["income_tax", "subsidy"],
                "reward_type" : "gdp_growth_stability",
                # "observation_types" : ["gdp", "bank_deposits"],
                "observation_types" : observation_types,
                "additional" : {
                    "alpha" : 100.0
                }
            },
            train_params = {
                "N_iter" : N_iter,
                "N_mu" : 1
            },
            adv_action_types = adv_action_types,
            gym_spaces_bounds = gym_spaces_bounds,
            # n_seeds = 3,
            n_seeds = 2,
            algo = algo,
            dir = dir
        )

    # # FIRST TRAINING CWR_0.1_0.9

    # dir_1 = os.path.join(base_dir, "CWR_0.1_0.9")
    # os.makedirs(dir_1, exist_ok = True)
    
    # training_1 = TrainPolicies(
    #     env_config = {
    #         "action_types" : ["income_tax", "subsidy"],
    #         "reward_type" : "gdp_growth_stability",
    #         "observation_types" : ["gdp", "bank_deposits"],
    #         "additional" : {
    #             "alpha" : 100.0
    #         }
    #     },
    #     train_params = {
    #         "N_iter" : 500,
    #         "N_mu" : 1
    #     },
    #     adv_action_types = ["consumption_wealth_ratio"],
    #     gym_spaces_bounds = {
    #         "act_consumption_wealth_ratio" : (0.1, 0.9)
    #     },
    #     n_seeds = 3,
    #     dir = dir_1
    # )

    # # SECOND TRAINING FIP_0.1_0.9

    # dir_2 = os.path.join(base_dir, "FIP_0.1_0.9")
    # os.makedirs(dir_2, exist_ok = True)
    
    # training_2 = TrainPolicies(
    #     env_config = {
    #         "action_types" : ["income_tax", "subsidy"],
    #         "reward_type" : "gdp_growth_stability",
    #         "observation_types" : ["gdp", "bank_deposits"],
    #         "additional" : {
    #             "alpha" : 100.0
    #         }
    #     },
    #     train_params = {
    #         "N_iter" : 500,
    #         "N_mu" : 1
    #     },
    #     adv_action_types = ["firm_invest_prob"],
    #     gym_spaces_bounds = {
    #         "act_firm_invest_prob" : (0.1, 0.9)
    #     },
    #     n_seeds = 3,
    #     dir = dir_2
    # )

    # # THIRD TRAINING MP_0.5_0.99

    # dir_3 = os.path.join(base_dir, "MP_0.5_0.99")
    # os.makedirs(dir_3, exist_ok = True)
    
    # training_3 = TrainPolicies(
    #     env_config = {
    #         "action_types" : ["income_tax", "subsidy"],
    #         "reward_type" : "gdp_growth_stability",
    #         "observation_types" : ["gdp", "bank_deposits"],
    #         "additional" : {
    #             "alpha" : 100.0
    #         }
    #     },
    #     train_params = {
    #         "N_iter" : 500,
    #         "N_mu" : 1
    #     },
    #     adv_action_types = ["memory_parameter"],
    #     gym_spaces_bounds = {
    #         "act_memory_parameter" : (0.5, 0.99)
    #     },
    #     n_seeds = 3,
    #     dir = dir_2
    # )

    # # FOURTH TRAINING CP_0.01_0.99

    # dir_2 = os.path.join(base_dir, "CP_0.01_0.99")
    # os.makedirs(dir_2, exist_ok = True)
    
    # training = TrainPolicies(
    #     env_config = {
    #         "action_types" : ["income_tax", "subsidy"],
    #         "reward_type" : "gdp_growth_stability",
    #         "observation_types" : ["gdp", "bank_deposits"],
    #         "additional" : {
    #             "alpha" : 100.0
    #         }
    #     },
    #     train_params = {
    #         "N_iter" : 500,
    #         "N_mu" : 1
    #     },
    #     adv_action_types = ["memory_parameter"],
    #     gym_spaces_bounds = {
    #         "act_memory_parameter" : (0.5, 0.99)
    #     },
    #     n_seeds = 3,
    #     dir = dir_2
    # )