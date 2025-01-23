from train import TrainPolicies

import os
import numpy as np

if __name__ == "__main__":

    # creating base directory
    # base_dir = "../results/test/exps_3"
    base_dir = "../results/exps"
    # base_dir = "../results/test/final_exps_3" # testing
    os.makedirs(base_dir, exist_ok = True)

    trainings = {
        "beta_-5.0_1.0" : {
            "adv_action_types" : ["beta"],
            "range" : (-5.0, 1.0)
        },
        # "beta_-3.0_-1.0" : {
        #     "adv_action_types" : ["beta"],
        #     "range" : (-3.0, -1.0)
        # },
        "alpha_-1.5_4.5" : {
            "adv_action_types" : ["alpha"],
            "range" : (-1.5, 4.5)
        },
        # "alpha_0.5_2.5" : {
        #     "adv_action_types" : ["alpha"],
        #     "range" : (0.5, 2.5)
        # },
        "gamma_-7.0_-1.0" : {
            "adv_action_types" : ["gamma"],
            "range" : (-7.0, -1.0)
        },
        # "gamma_-5.0_-3.0" : {
        #     "adv_action_types" : ["gamma"],
        #     "range" : (-5.0, -3.0)
        # },
        # "CWR_0.01_0.99" : {
        #     "adv_action_types" : ["consumption_wealth_ratio"],
        #     "range" : (0.01, 0.99)
        # },
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
        # "MP_0.01_0.99" : {
        #     "adv_action_types" : ["memory_parameter"],
        #     "range" : (0.01, 0.99)
        # }
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

        training = TrainPolicies(
            env_config = {"T" : 100},
            train_params = {
                "N_iter" : 4000,
                "N_mu" : 1
            },
            adv_action_types = adv_action_types,
            gym_spaces_bounds = gym_spaces_bounds,
            n_seeds = 2,
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