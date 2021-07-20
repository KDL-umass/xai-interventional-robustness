from all.experiments import SlurmExperiment, run_experiment
from all.environments import AtariEnvironment
from all.presets import atari
import argparse
from all.presets.atari import (
    a2c,
    dqn,
    vac,
    vpg,
    vsarsa,
    vqn
)

from envs.wrappers.space_invaders_features.all_toybox_wrapper import ToyboxEnvironment

parser = argparse.ArgumentParser(description="Run an Atari benchmark for interventional robustness.")
parser.add_argument("--env", default = "SpaceInvaders", help="Name of the Atari game (e.g. Pong).")
# passed as a list # parser.add_argument(
#     "agent", help="Name of the agent (e.g. dqn). See presets for available agents."
# )
parser.add_argument(
    "--device",
    default="cuda",
    help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
)
parser.add_argument(
    "--frames", type=int, default=40e6, help="The number of training frames."
)
parser.add_argument(
    "--render", action="store_true", default=False, help="Render the environment."
)
parser.add_argument(
    "--logdir", default='runs', help="The base logging directory."
)
parser.add_argument(
    "--writer", default='tensorboard', help="The backend used for tracking experiment metrics."
)
# parser.add_argument('--hyperparameters', default=[], nargs='*')
parser.add_argument('--toybox', action = "store_true", default=False, help = "Import environment from Toybox?")
args = parser.parse_args()

# print all the arguments 
for arg in vars(args):
    print(arg, getattr(args, arg))


def main():
    device = args.device 
    if args.toybox:
        env = ToyboxEnvironment('SpaceInvadersToybox', device=device)
    else:
        env = AtariEnvironment(args.env, device=device)
    agents = [
        a2c.device(device),
        # dqn.device(device),
    ]
    if device == "cuda":
        SlurmExperiment(agents, env, args.frames, render = args.render, logdir = args.logdir, writer=args.writer, sbatch_args={
            'partition': '1080ti-long'
        })
    else:
        run_experiment(agents, env, args.frames, render = args.render, logdir=args.logdir,
        writer=args.writer)

if __name__ == "__main__":
    main()
