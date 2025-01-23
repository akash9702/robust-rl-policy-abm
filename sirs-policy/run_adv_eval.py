from robust_eval import EvaluateRobustness

import os
import numpy as np

from stable_baselines3 import PPO

if __name__ == "__main__":
    
    base_dir = "../results/exps"

    trainings = [
        "beta_-5.0_1.0",
        "alpha_-1.5_4.5",
        "gamma_-7.0_-1.0"
    ]

    adv_action_types_dict = {
        "beta_-5.0_1.0" : "beta",
        "alpha_-1.5_4.5" : "alpha",
        "gamma_-7.0_-1.0" : "gamma"
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
            n_train_seeds = 2
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

