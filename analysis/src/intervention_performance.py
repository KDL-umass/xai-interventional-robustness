from tabnanny import check
import pandas as pd 
import numpy as np 
import argparse, os, glob
from analysis.checkpoints import all_checkpoints, checkpoints as paper_checkpoints

def main(env_name, fam, intervention=-1, checkpoint=None):

    if checkpoint is not None:
        assert checkpoint in all_checkpoints, "checkpoint not available"
        checkpoints = [checkpoint]
    else:
        checkpoints = paper_checkpoints
    df_list = []
    for checkpoint in checkpoints:
        dir = f"storage/results/intervention_js_div/{env_name}/{fam}/11_agents/30_states/trajectory/check_{checkpoint}/"
        number = {'Amidar':69, 'SpaceInvaders':88, 'Breakout':38}
        file = dir + f"{number[env_name]}_interventions.txt"
        d = pd.read_csv(file, sep=' ', skiprows=0, header=0, names = ["agent","state","intervention","js_div"])
        d = d.astype({"intervention":int,"state":int})
        d = d[d["state"]==0]
        d = d.drop(columns=['agent','state'])
        d["frame"] = checkpoint           
        df_list.append(d)
    

    df_r = pd.concat(df_list, ignore_index=True)
    
    df_list = []
    dir = f"storage/results/intervention_performance/{env_name}/{fam}"
    loadfiles = glob.glob(dir + "/*.txt")
    for file in loadfiles: 
        d = pd.read_csv(file, sep=',')
        if d.empty: 
            print("Failed: ", file)
        elif len(d)!=55: 
            if fam == "c51": 
                d = d.loc[d["agent"].isin([1,2,3,4,5,6,7,8,9,10,11])]
                df_list.append(d)
            elif fam == "ddqn": 
                d = d.loc[d["agent"].isin([1,2,3,4,5,6,7,8,9,10,11])]
                df_list.append(d)
            else: 
                print("Length: ", len(d), " File: ", file)
        else:
            df_list.append(d)            

    df_p = pd.concat(df_list, ignore_index=True)
    df = df_p.groupby(["intervention","frame"]).mean()
    df = df.rename(columns = {"mean":"mean_performance", "std":"mean_std"})
    df = df.drop(columns = ['agent'])

    df = pd.merge(df, df_r, how="left", on=["intervention","frame"])
    file = f"storage/results/intervention_performance_results/{env_name}_{fam}_intvperf.csv"
    df.to_csv(file)

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

    args = parser.parse_args()

    main(args.env[0], args.family[0])
