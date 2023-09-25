import numpy as np
import torch
import tensorflow as tf
import os
import gym

from all.presets.atari import c51, rainbow, a2c, dqn, vsarsa, vqn, ppo, ddqn
from all.experiments.single_env_experiment import SingleEnvExperiment
from all.experiments.parallel_env_experiment import ParallelEnvExperiment
from envs.wrappers.paths import get_num_interventions
from envs.wrappers.all_toybox_wrapper import (
    ToyboxEnvironment,
    customAmidarResetWrapper,
    customBreakoutResetWrapper,
    customSpaceInvadersResetWrapper,
)

def test_experiment(
        agent, 
        loadfile, 
        env,
        frames,
        logdir='runs',
        quiet=False,
        render=False,
        test_episodes=100,
        write_loss=True,
        writer="tensorboard",
    ):

    env.seed(0)
    preset = torch.load(loadfile)
    make_experiment = get_experiment_type(preset)
    print("make_experiment")
    print(make_experiment)
    experiment = make_experiment(
        preset,
        env,
        train_steps=frames,
        logdir=logdir,
        quiet=quiet,
        render=render,
        write_loss=write_loss,
        writer=writer,
    )
    experiment.test(episodes=test_episodes)

# take one trained agent 
# get final hidden layer activations for vanilla states 
def get_final_hidden_activations(data):
    return activations

# Do the same thing as in #1 but on a set of intervened envs 

# for each of the intervened activations, look at mean distance to the 10 nearest vanilla states
def activation_dist(vactivations, activation):
    return dists

# get IR for the intervened state and 10 nearest vanilla states


if __name__=='__main__':
    loadfile = "/gypsum/work1/jensen/pboddavarama/xai-interventional-robustness/storage/models/Amidar/a2c/a2c_5ef3d29_2021-12-18_16:59:00_572126/preset10000000.pt"
    n_agents = 11
    nstates = 30
    cesampling = True
    checkpoint = 10000000
    env = "Amidar"
    folder = "intervention_ce" if cesampling else "intervention_action_dists"
    fam = "a2c" 
    nintv = get_num_interventions(env)

    dir = f"/gypsum/work1/jensen/pboddavarama/xai-interventional-robustness/storage/results/{folder}/{env}/{fam}/{n_agents}_agents/{nstates}_states/trajectory/check_{checkpoint}"

    vdata = np.loadtxt(dir + f"/vanilla.txt")
    data = np.loadtxt(dir + f"/{nintv}_interventions.txt")
    print("loaded txt files")

    agent = a2c.device("cuda")
    custom_wrapper = customAmidarResetWrapper(0, -1, 3)
    env = ToyboxEnvironment(
            env_name + "Toybox", device="cuda", custom_wrapper=custom_wrapper
        )

    test_experiment(agent, loadfile, env, checkpoint)
    #vactivations = get_final_hidden_activations(vdata)
    #activations = get_final_hidden_activations(data)

    # base_experiment(loadfile, "/home/kavery_umass_edu/xai-interventional-robustness/storage/results", save_activations=True)
    # print("base experiment")


    # for activation in activations: 
    #     dists = activation_dist(vactivations, activation)
    #     dists = dists[0:10]
    #     #ir 

