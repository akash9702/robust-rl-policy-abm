from robust_eval import EvaluateRobustness

import os
import numpy as np

from stable_baselines3 import PPO

if __name__ == "__main__":
    
    base_dir = "../results/exps"

    # trainings = {
    #     "beta_-5.0_1.0" : {
    #         "adv_action_types" : ["beta"],
    #         "range" : (-5.0, 1.0)
    #     },
    #     # "beta_-3.0_-1.0" : {
    #     #     "adv_action_types" : ["beta"],
    #     #     "range" : (-3.0, -1.0)
    #     # },
    #     "alpha_-1.5_4.5" : {
    #         "adv_action_types" : ["alpha"],
    #         "range" : (-1.5, 4.5)
    #     },
    #     # "alpha_0.5_2.5" : {
    #     #     "adv_action_types" : ["alpha"],
    #     #     "range" : (0.5, 2.5)
    #     # },
    #     "gamma_-7.0_-1.0" : {
    #         "adv_action_types" : ["gamma"],
    #         "range" : (-7.0, -1.0)
    #     },
    # }

    trainings = [
        "beta_-5.0_1.0",
        "alpha_-1.5_4.5",
        "gamma_-7.0_-1.0"
    ]

    shifts = {
        "beta_-5.0_1.0" : {
            "name" : "beta_-5.0_1.0",
            "shift" : {
                "param" : "beta",
                "range" : (-5.0, 1.0),
                "n_values" : 10,
                "logspace" : False
            }
        },
        "alpha_-2.0_4.0" : {
            "name" : "alpha_-2.0_4.0",
            "shift" : {
                "param" : "alpha",
                "range" : (-2.0, 4.0),
                "n_values" : 10,
                "logspace" : False
            }
        },

        "gamma_-7.0_-1.0" : {
            "name" : "gamma_-7.0_-1.0",
            "shift" : {
                "param" : "gamma",
                "range" : (-7.0, -1.0),
                "n_values" : 10,
                "logspace" : False
            }
        }
    }

    # shifts = {
    #     "beta_-5.0_1.0" : {
    #         "name" : "beta_-5.0_1.0",
    #         "shift" : {
    #             "param" : "beta",
    #             "range" : (-5.0, 1.0),
    #             "n_values" : 3,
    #             "logspace" : False
    #         }
    #     },
    #     "alpha_-2.0_4.0" : {
    #         "name" : "alpha_-2.0_4.0",
    #         "shift" : {
    #             "param" : "alpha",
    #             "range" : (-2.0, 4.0),
    #             "n_values" : 3,
    #             "logspace" : False
    #         }
    #     },

    #     "gamma_-7.0_-1.0" : {
    #         "name" : "gamma_-7.0_-1.0",
    #         "shift" : {
    #             "param" : "gamma",
    #             "range" : (-7.0, -1.0),
    #             "n_values" : 3,
    #             "logspace" : False
    #         }
    #     }
    # }

    for training in trainings:

        dir = os.path.join(base_dir, training)

        eval = EvaluateRobustness(
            dir = dir,
            n_seeds = 10,
            n_train_seeds = 2
        )

        for shift_name, shift_args in shifts.items():

            eval.add_param_shift_one(
                **shift_args
            )
            
            # eval.plot_existing_datasets_one_shift(**shift_args)