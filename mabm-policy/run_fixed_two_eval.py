from robust_eval import EvaluateRobustness

import os
import numpy as np

from stable_baselines3 import PPO

if __name__ == "__main__":
    
    base_dir = "../results/exps"

    trainings = [
        "CWR_0.1_0.9",
        "MP_0.5_0.99",
        "CP_0.1_0.9",
        "LP_0.1_0.9",
    ]

    shifts = {
        "FIP_0.2_1.0_MP_0.0_1.0" : {
            "name" : "FIP_0.2_1.0_MP_0.0_1.0",
            "shifts" : [
                {
                    "param" : "firm_invest_prob",
                    "n_values" : 9,
                    "range" : (0.2, 1.0),
                    "logspace" : False,
                },
                {
                    "param" : "memory_parameter",
                    "n_values" : 11,
                    "range" : (0.0, 1.0),
                    "logspace" : False,
                    "values" : np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.93, 0.96, 0.99, 1.0])
                }
            ]
        },
        "CP_0.15_0.85_MP_0.0_1.0" : {
            "name" : "CP_0.15_0.85_MP_0.0_1.0",
            "shifts" : [
                {
                    "param" : "capital_prod",
                    "n_values" : 11,
                    "range" : (0.15, 0.85),
                    "logspace" : False
                },
                {
                    "param" : "memory_parameter",
                    "n_values" : 11,
                    "range" : (0.0, 1.0),
                    "logspace" : False,
                    "values" : np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.93, 0.96, 0.99, 1.0])
                }
            ]
        },
        "FIP_0.2_1.0_LP_0.4_0.8" : {
            "name" : "FIP_0.2_1.0_LP_0.4_0.8",
            "shifts" : [
                {
                    "param" : "firm_invest_prob",
                    "n_values" : 10,
                    "range" : (0.2, 1.0),
                    "logspace" : False,
                },
                {
                    "param" : "labour_prod",
                    "n_values" : 9,
                    "range" : (0.4, 0.8),
                    "logspace" : False
                }
            ]
        }
    }

    for training in trainings:

        dir = os.path.join(base_dir, training)

        eval = EvaluateRobustness(
            dir = dir,
            n_seeds = 2,
            n_train_seeds = 3
        )

        for shift_name, shift_args in shifts.items():

            # eval.add_param_shift_two(
            #     **shift_args
            # )

            eval.plot_existing_datasets_two_shifts(
                **shift_args
            )