from robust_eval import EvaluateRobustness

import os
import numpy as np

from stable_baselines3 import PPO

if __name__ == "__main__":
    
    base_dir = "../results/exps"

    trainings = [
        "CWR_0.1_0.9",
        "FIP_0.1_0.9",
        "MP_0.5_0.99",
        "CP_0.1_0.9",
        "LP_0.1_0.9",
        "CWR_0.01_0.99",
        "CWR_0.01_0.3",
        "FIP_0.1_0.5",
        "CP_0.2_0.8",
        "LP_0.5_0.75",
        "MP_0.01_0.99"
    ]

    shifts = {
        "CWR_0.1_1.0" : {
            "name" : "CWR_0.1_1.0",
            "shift" : {
                "param" : "consumption_wealth_ratio",
                "n_values" : 9,
                "range" : (0.1, 1.0),
                "logspace" : False
            }
        },
        "MP_0.0_1.0" : {
            "name" : "MP_0.0_1.0",
            "shift" : {
                "param" : "memory_parameter",
                "n_values" : 10,
                "range" : (0.0, 1.0),
                "logspace" : False,
                "values" : np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 0.99, 1.0])
            }
        },
        "FIP_0.1_1.0" : {
            "name" : "FIP_0.1_1.0",
            "shift" : {
                "param" : "firm_invest_prob",
                "n_values" : 9,
                "range" : (0.1, 1.0),
                "logspace" : False
            }
        },
        "LP_0.4_0.8" : {
            "name" : "LP_0.4_0.8",
            "shift" : {
                "param" : "labour_prod",
                "n_values" : 9,
                "range" : (0.4, 0.8),
                "logspace" : False
            }
        },
        "CP_0.15_0.85" : {
            "name" : "CP_0.15_0.85",
            "shift" : {
                "param" : "capital_prod",
                "n_values" : 11,
                "range" : (0.15, 0.85),
                "logspace" : False
            }
        }
    }

    for training in trainings:

        dir = os.path.join(base_dir, training)

        eval = EvaluateRobustness(
            dir = dir,
            n_seeds = 10,
            n_train_seeds = 3
        )

        for shift_name, shift_args in shifts.items():

            # eval.add_param_shift_one(
            #     **shift_args
            # )
            
            eval.plot_existing_datasets_one_shift(**shift_args)