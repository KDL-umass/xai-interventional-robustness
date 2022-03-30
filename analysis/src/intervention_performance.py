from tabnanny import check
import pandas as pd 
import numpy as np 
import argparse, os, glob

def main(env_name, fam, intervention=-1, checkpoint=None):
    robustness_score = None
    
    if intervention is not None and checkpoint is not None:   
        returns_file = f"storage/results/performance/{env_name}/{fam}/returns_intv_{intervention}.txt"
        f = pd.read_csv(returns_file, sep=',')  
        print(f.groupby('frame').mean())
    else:
        df_list = []
        dir = f"storage/results/performance/{env_name}/{fam}"
        loadfiles = glob.glob(dir + "/*.txt")
        for file in loadfiles: 
            df_list.append(pd.read_csv(file, sep=','))

        df = pd.concat(df_list, ignore_index=True)
        print(df.groupby('intervention').mean())

    return robustness_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse intervention performance of a single agent family."
    )
    parser.add_argument(
        "--env",
        nargs=1,
        type=str,
        help="environment name: SpaceInvaders, Amidar, or Breakout",
    )
    parser.add_argument(
        "--family",
        nargs=1,
        type=str,
        help="Agent family:  a2c,c51, dqn, ddqn, ppo, rainbow, vsarsa, vqn",
    )

    parser.add_argument(
        "--intervention",
        nargs=1,
        type=int,
        help="(optional) intervention number for each environment and family"
    )

    # Not using checkpoints, automatically run all experiments for all the checkpoints mentioned in the main() function
    parser.add_argument(
        "--checkpoint",
        nargs=1,
        type=int,
        help="Checkpoint to load",
    )

    args = parser.parse_args()

    if args.intervention is not None and args.checkpoint is not None:
        main(args.env[0], args.family[0], args.intervention[0], args.checkpoint[0])
    else:
        main(args.env[0], args.family[0])