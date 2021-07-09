from space_invaders_wrapper.space_invaders_feature_vec_wrapper import (
    SpaceInvadersFeatureVecWrapper,
)
import gym
from space_invaders_wrapper.wrappers import wrap_space_env
import torch
import numpy as np
import random
import os, sys

from agents.ddt_agent import DDTAgent, load_ddt

from IQN_agent import wrapper
from IQN_agent.agent import IQN_Agent

import torch.multiprocessing as mp

from .video_utils import *
from .space_invaders_interventions import get_env_list
from .space_invaders_reset_wrapper import wrap_space_env_reset


def load_agent(agent_name, env, seed):
    if agent_name == "ddt":
        ddt_path = (
            "./modelpth/ddt_saved_models/29th_ddt_space_GPU_8_leaves_actor.pth.tar"
        )
        fda = load_ddt(ddt_path)
        model = DDTAgent(bot_name="crispytester")
        model.action_network = fda
        model.value_network = fda

    elif agent_name == "cnn":
        eval_env = wrapper.wrap_env(env)
        action_size = eval_env.action_space.n
        state_size = eval_env.observation_space.shape

        model = IQN_Agent(
            state_size=state_size,
            action_size=action_size,
            network="iqn",
            munchausen=0,
            layer_size=512,
            n_step=1,
            BATCH_SIZE=32,
            BUFFER_SIZE=10000,
            LR=0.00025,
            TAU=1e-3,
            GAMMA=0.99,
            N=8,
            worker=1,
            device="cuda:0" if torch.cuda.is_available() else torch.device("cpu"),
            seed=seed,
        )
        model.qnetwork_local.load_state_dict(
            torch.load("iqn2.pth", map_location=torch.device("cpu"))
        )
    elif agent_name == "random":
        model = None
    return model


def run_episode(
    agent_name, env, seed, env_name, intv, lives, save_images=False, save_actions=False
):
    """Run episode with provided arguments."""

    agent = load_agent(agent_name, env, seed)
    if agent_name == "ddt":
        env = wrap_space_env(env, buffer_size=4, intv=intv, lives=lives)

    elif agent_name == "cnn":
        env = wrapper.wrap_env(env, intv=intv, lives=lives)

    done = False

    torch.manual_seed(seed)
    env.seed(seed)
    np.random.seed(seed)
    env.action_space.seed(seed)
    random.seed(seed)

    state = env.reset()  # assumes intervention wrapper

    t = 0
    if save_images:
        dir_path = get_image_path(env_name, seed, agent_name)
        record_image(t, env, env_name, seed, agent_name)

    # Create directory to store the trajectories for actions
    if save_actions:
        action_path = "interventions/results/action_trajectories/{}/{}/{}/".format(
            env_name, seed, agent_name
        )
        if not os.path.exists(action_path):
            os.makedirs(action_path)

        # At the beginning of each call, erase the existing contents 
        f = open(action_path + "{}.txt".format(agent_name), "w")
        f.close()


    rewards = 0
    while not done:
        t = t + 1
        if agent_name == "ddt":
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
        elif agent_name == "cnn":
            action = agent.act(np.expand_dims(state, axis=0), 0.001, eval=True)
            state, reward, done, _ = env.step(action[0].item())
        elif agent_name == "random":
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
        else:
            print("Error: Unknown agent specified.")
            break

        rewards += reward
        if save_images:
            env.toybox.save_frame_image(dir_path + "{}.png".format(t))
        if save_actions:
            with open(action_path + "{}.txt".format(agent_name), "a+") as f:
                if agent_name == "cnn":
                    record_action = str(action[0])
                else:
                    record_action = str(action)
                f.write(record_action)
        if done:
            break

    if save_images:
        dir_path = get_image_path(env_name, seed, agent_name)
        video_path = "interventions/results/videos/"
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        make_videos(
            dir_path,
            video_path + "{}_{}_{}.mp4".format(env_name, seed, agent_name),
        )

    print("Episode reward:", rewards)
    print("Time steps: ", t)

    if save_actions:
        with open(action_path + "{}.txt".format(agent_name), "a+") as f:
            f.write("\n" + str(rewards))
    return rewards


def want_feature_vec(agent_name):
    return True if agent_name == "ddt" else False


def run_trial(agent_name, env_number, seed, vanilla, lives):
    print(f"Env: {env_number}, Seed: {seed}")
    env_list = get_env_list(want_feature_vec(agent_name), vanilla, lives)
    env = env_list[env_number]

    if vanilla:
        intv = -1
    else:
        intv = env_number

    return run_episode(agent_name, env, seed, env_number, intv, lives)


def evaluate(agent_name, num_trials, vanilla, parallel, lives, save_images=False):
    """Evaluate agent on list of environments created from arguments `agent_name`, `vanilla`, and `lives`."""
    env_list = get_env_list(want_feature_vec(agent_name), vanilla, lives)

    if parallel:
        args = [
            (agent_name, e, t, vanilla, lives)
            for e in range(len(env_list))
            for t in range(num_trials)
        ]

        with mp.Pool(int(mp.cpu_count() * 0.9)) as pool:
            reward_list = pool.starmap(run_trial, args)
        return np.reshape(reward_list, (len(env_list), num_trials))

    else:
        reward_list = np.zeros((len(env_list), num_trials))
        for (e, env) in enumerate(env_list):
            for t in range(num_trials):
                print(
                    f"Env {e+1}/{len(env_list)}, Trial {t+1}/{num_trials}, aka {round(((e * num_trials + t)+1) / (len(env_list) * num_trials), 2)}%"
                )
                if vanilla:
                    intv = -1
                else:
                    intv = e
                reward = run_episode(
                    agent_name, env, t, intv, intv, lives, save_images=save_images
                )
                reward_list[e, t] = reward

        return reward_list  # returning full reward list for statistical analysis


def store_action_trajectories():
    """
    Function to store the action trajectories for seeds 0-29 for different interventions.
    """
    # Store action trajectories to compare the resulting trajectories with and without interventions.
    for s in range(30):
        for agent_name in ["cnn", "ddt", "random"]:
            print("Agent name: ", agent_name)
            save_images = False
            save_actions = True
            seed = s
            lives = 1

            # Without interventions
            vanilla = True
            intv = -1
            env_list = get_env_list(want_feature_vec(agent_name), vanilla, lives)
            env = env_list[intv]
            env_name = intv

            print("Seed: ", seed, " Vanilla: ", vanilla)

            run_episode(
                agent_name,
                env,
                seed,
                env_name,
                intv=-1,
                lives=lives,
                save_images=save_images,
                save_actions=save_actions,
            )

            print("######################################")

            # With interventions
            vanilla = False
            env_list = get_env_list(want_feature_vec(agent_name), vanilla, lives)
            for intv in range(87):
                env = env_list[intv]
                env_name = intv

                print("Seed: ", seed, " Vanilla: ", vanilla, "Intervention: ", intv)

                run_episode(
                    agent_name,
                    env,
                    seed,
                    env_name,
                    intv=intv,
                    lives=lives,
                    save_images=save_images,
                    save_actions=save_actions,
                )

            print("######################################")
            print("######################################")


if __name__ == "__main__":
    parallel = False
    num_trials = 30
    lives = 1
    save_images = False

    for agent_name in ["cnn", "random", "ddt"]:
        # Interventions
        performance_matrix = evaluate(
            agent_name, num_trials, False, parallel, lives, save_images=save_images
        )
        np.savetxt(
            f"./interventions/results/{agent_name}_performance_lives_{lives}.txt",
            performance_matrix,
        )

        # Vanilla
        performance_matrix = evaluate(
            agent_name, num_trials, True, parallel, 1, save_images=False
        )
        np.savetxt(
            f"./interventions/results/{agent_name}_performance_vanilla_lives_{lives}.txt",
            performance_matrix,
        )

    # # visualize play for an agent for given intervention and seed.
    # agent_name = "cnn"
    # save_images = True
    # vanilla = False
    # seed = 2
    # intv = 0
    # lives = 1
    # env_list = get_env_list(want_feature_vec(agent_name), vanilla, lives)
    # env = env_list[intv]
    # env_name = intv
    # run_episode(
    #     agent_name,
    #     env,
    #     seed,
    #     env_name,
    #     intv,
    #     lives,
    #     save_images=save_images,
    # )

    # Action trajectories
    # store_action_trajectories()

    """
    # Print out action sequences for DDT agent under the "shift agent" intervention
    # Single instance
    intv = 70
    seed = 0
    vanilla = False
    agent_name = "ddt"
    lives = 1
    save_actions = True
    save_images = False
    
    env_list = get_env_list(want_feature_vec(agent_name), vanilla, lives)
    env = env_list[intv]
    env_name = intv
    
    print("Intervention: ", intv)
    print("##########################")
    run_episode(agent_name, env, seed, env_name, intv, lives, save_images=save_images, save_actions=save_actions)
    print("##########################")

    # Vanilla trial; change intervention and flag
    intv = -1
    vanilla = True
    
    env_list = get_env_list(want_feature_vec(agent_name), vanilla, lives)
    env = env_list[intv]
    env_name = intv

    print("Intervention: NONE")
    print("##########################")
    run_episode(agent_name, env, seed, env_name, intv, lives, save_images=save_images, save_actions=save_actions)
    print("##########################")
    """