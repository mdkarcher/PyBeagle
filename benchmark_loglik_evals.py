import phyloinfer as pinf
import timeit
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from loglikelihood import *
from models import *


def cheng_lik_list_eval(tree_list, branch_list, L, model=JC):
    result = []
    for tree, branch in zip(tree_list, branch_list):
        # branch = pinf.branch.get(tree)
        llik = pinf.Loglikelihood.phyloLoglikelihood(tree, branch, model.D, model.U, model.U_inv, model.pi, L)
        result.append(llik)
    return result


def bg_lik_list_eval(tree_list, bg_init, model=JC):
    bg_instance, tip_name_to_address = bg_init
    result = []
    for tree in tree_list:
        llik = loglikelihood_beagle_evaluate(bg_instance, tree, tip_name_to_address, scaling=True)
        result.append(llik)
    return result


def method_tester(method_invocation_dict, variable_list_dict, repeat=5, verbose=False):
    variables = tuple(variable_list_dict)
    # columns = [*variables, "method", "time"]
    # df = pd.DataFrame(columns=columns)
    df_dict = defaultdict(list)
    for method in method_invocation_dict:
        invocation = method_invocation_dict[method]
        if verbose:
            print(f"### Method '{method}' with invocation '{invocation}':")
        variable_combinations = itertools.product(*(variable_list_dict[variable] for variable in variables))
        for comb in variable_combinations:
            setup = ""
            df_dict["method"].append(method)
            for variable, value in zip(variables, comb):
                setup += f"{variable} = {value};"
                df_dict[variable].append(value)
            if verbose:
                print(f"Setup code: '{setup}'")
            times = timeit.repeat(stmt=invocation, setup=setup, repeat=repeat, number=1, globals=globals())
            time = min(times)
            if verbose:
                print(f"Min run time: {time}")
            df_dict["time"].append(time)
    df = pd.DataFrame.from_dict(df_dict)
    return df


# Experiment 0

n_sims = 10
n_tips = 200
n_sites = 1000
model = JC

tree_list = []
branch_list = []
true_tree = pinf.tree.create(n_tips, branch='random')
data = pinf.data.treeSimu(true_tree, model.D, model.U, model.U_inv, model.pi, n_sites)
L = pinf.Loglikelihood.initialCLV(data)
bg_init = loglikelihood_beagle_init(data, model, scaling=True)
for _ in range(n_sims):
    tree = pinf.tree.create(n_tips, branch='random')
    tree_list.append(tree)
    branch = pinf.branch.get(tree)
    branch_list.append(branch)

cheng_lliks = cheng_lik_list_eval(tree_list, branch_list, L, model)
bg_lliks = bg_lik_list_eval(tree_list, bg_init, model)
np.round(np.array(cheng_lliks) - np.array(bg_lliks), 9)

# %timeit cheng_lik_list_eval(tree_list, branch_list, L, model)
# %timeit bg_lik_list_eval(tree_list, bg_init, model)

# Experiment 1

n_sims = 10
n_tips_list = [16, 32, 64, 128, 256, 512, 1024]
n_sites_list = [16, 32, 64, 128, 256, 512, 1024]
model = JC

# df = pd.DataFrame(columns=["n_tips", "n_sites", "method", "time"])
df_dict = defaultdict(list)
results = {}
for n_tips, n_sites in itertools.product(n_tips_list, n_sites_list):
    tree_list = []
    branch_list = []
    true_tree = pinf.tree.create(n_tips, branch='random')
    data = pinf.data.treeSimu(true_tree, model.D, model.U, model.U_inv, model.pi, n_sites)
    L = pinf.Loglikelihood.initialCLV(data)
    bg_init = loglikelihood_beagle_init(data, model, scaling=True)
    for _ in range(n_sims):
        tree = pinf.tree.create(n_tips, branch='random')
        tree_list.append(tree)
        branch = pinf.branch.get(tree)
        branch_list.append(branch)
    cheng_times = timeit.repeat("cheng_lik_list_eval(tree_list, branch_list, L, model)", repeat=7, number=1, globals=globals())
    df_dict["n_tips"].append(n_tips)
    df_dict["n_sites"].append(n_sites)
    df_dict["method"].append("cheng")
    df_dict["time"].append(min(cheng_times))
    bg_times = timeit.repeat("bg_lik_list_eval(tree_list, bg_init, model)", repeat=7, number=1, globals=globals())
    df_dict["n_tips"].append(n_tips)
    df_dict["n_sites"].append(n_sites)
    df_dict["method"].append("beagle")
    df_dict["time"].append(min(bg_times))
    # df = df.append({"n_tips": n_tips, "n_sites": n_sites, "method": "cheng", "time": min(cheng_times)},
    #                ignore_index=True)
    # df = df.append({"n_tips": n_tips, "n_sites": n_sites, "method": "beagle", "time": min(bg_times)}, ignore_index=True)
    results[n_tips, n_sites] = (min(cheng_times), min(bg_times))
df = pd.DataFrame.from_dict(df_dict)

for n_tips, n_sites in results:
    cheng_time, bg_time = results[n_tips, n_sites]
    print(f"{n_tips} tips, {n_sites} sites: Cheng {cheng_time:.3g}s, Beagle {bg_time:.3g}s, {(cheng_time/bg_time - 1)*100:.4g}% speedup")

df[df.method == "cheng"].pivot(index="n_tips", columns="n_sites", values="time")
df[df.method == "beagle"].pivot(index="n_tips", columns="n_sites", values="time")


