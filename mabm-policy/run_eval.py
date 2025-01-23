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

    adv_action_types_dict = {
        "CWR_0.1_0.9" : "consumption_wealth_ratio",
        "FIP_0.1_0.9" : "firm_invest_prob",
        "MP_0.5_0.99" : "memory_parameter",
        "CP_0.1_0.9": "capital_prod",
        "LP_0.1_0.9" : "labour_prod",
        "CWR_0.01_0.99" : "consumption_wealth_ratio",
        "CWR_0.01_0.3" : "consumption_wealth_ratio",
        "FIP_0.1_0.5" : "firm_invest_prob",
        "CP_0.2_0.8": "capital_prod",
        "LP_0.5_0.75" : "labour_prod",
        "MP_0.01_0.99" : "memory_parameter"
    }

    adv_models = {}

    for training in trainings:

        dir = os.path.join(base_dir, training)

        adv_model_path = os.path.join(dir, f"42/RARL/saved/adv_model.zip")
        adv_model = PPO.load(adv_model_path)

        adv_models[training] = adv_model

    for training in trainings:

        dir = os.path.join(base_dir, training)

        eval = EvaluateRobustness(
            dir = dir,
            n_seeds = 10,
            n_train_seeds = 3
        )

        for adv_training in trainings:

            # eval.add_adversarial_model(
            #     name = adv_training,
            #     adv_model = adv_models[adv_training],
            #     adv_action_types = [adv_action_types_dict[adv_training]]
            # )

            eval.plot_existing_adv(
                name = adv_training,
                adv_model = adv_models[adv_training],
                adv_action_types = [adv_action_types_dict[adv_training]]
            )

