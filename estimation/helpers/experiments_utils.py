"""Wrapper for the experiment."""

from functools import partial
import numpy as np
import pandas as pd
import pyximport
import multiprocess as mp
import torch.multiprocessing as tmp
from .models import CM
from .utils import reduce_list
from other_models.direct_lingam.simulate_data import simulate_data
pyximport.install(inplace=True)

def run_experiment(seed,
                   highest_l=1,
                   adjacency_obs=None,
                   adjacency_latent=None,
                   noise_distribution='gamma',
                   model_class = CM,
                   methods = None,
                   location_interest = ((2, 1), )):
    """run the experiment using the given parameters"""

    print(f"Running experiment with seed {seed}\n")
    samples_sizes = [10**n for n in range(1, 8, 2)]
    n_max = int(max(samples_sizes))


    np.random.seed(seed)
    x, true_b = simulate_data(n_max,
                              noise_distribution=noise_distribution,
                              Lambda=adjacency_obs,
                              Gamma=adjacency_latent,
                              permute_order=False)

    causal_effect_true = reduce_list([true_b[x] for x in location_interest])
    # Perform the experiment


    estimates_dict = {methods[i]: [] for i in range(len(methods))}


    for n in samples_sizes:
        x_sample = x[:int(n), :]
        x_sample = np.asfortranarray(x_sample)
        x_sample = x_sample - np.mean(x_sample, axis=0)

        model = model_class(x_sample, highest_l=highest_l)
        if 'Cumulant' in methods:
            estimates_dict['Cumulant'].append(model.estimate_effect())
        if 'Cumulant Minimization' in methods:
            estimates_dict['Cumulant Minimization'].append(model.estimate__effect_minimization())
        if 'Cross Moment' in methods:
            estimates_dict['Cross Moment'].append(model.estimate_effect_cross_moment())
        if 'Relvlingam' in methods:
            _, b_estimate = model.fit(x_sample)
            relv_estimate = reduce_list([b_estimate[x] for x in location_interest])
            estimates_dict['Relvlingam'].append(relv_estimate)
        if 'Min Norm' in methods:
            estimates_dict['Min Norm'].append(model.estimate_effect_min_norm())


    # Create a DataFrame for the current seed
    seed_df = pd.DataFrame({
        'Seed': [seed] * len(samples_sizes),
        'Sample Size': samples_sizes,
        'True Effect': [causal_effect_true] * len(samples_sizes),
    })

    for method in methods:

        seed_df[method + ' Estimate'] = estimates_dict[method]

    print(f"Experiment with seed {seed} done\n")
    return seed_df

def run_experiment_grica(seed,
                         highest_l=1,
                         adjacency_obs=None,
                         adjacency_latent=None,
                         noise_distribution='gamma',
                         model_class = CM):

    print(f"Running GRICA experiment with seed {seed}\n")
    samples_sizes = [10**n for n in range(1, 8, 2)]
    n_max = int(max(samples_sizes))


    np.random.seed(seed)
    x, true_b = simulate_data(n_max,
                              noise_distribution=noise_distribution,
                              Lambda=adjacency_obs,
                              Gamma=adjacency_latent,
                              permute_order=False)

    causal_effect_true = true_b[2, 1]

    # Perform the experiment

    grica_causal_effect_estimate = []

    for n in samples_sizes:
        x_sample = x[:int(n), :]
        x_sample = np.asfortranarray(x_sample)
        x_sample = x_sample - np.mean(x_sample, axis=0)

        model = model_class(x_sample, highest_l=highest_l)

        if n <= 10**5:
            ce_estimate = model.estimate_effect_grica()
        grica_causal_effect_estimate.append(ce_estimate)



    # Create a DataFrame for the current seed
    seed_df = pd.DataFrame({
        'Seed': [seed] * len(samples_sizes),
        'Sample Size': samples_sizes,
        'True Effect': [causal_effect_true] * len(samples_sizes),
        'GRICA Estimate': grica_causal_effect_estimate
    })

    print(f"GRICA experiment with seed {seed} done\n")
    return seed_df

def parallel_simulation(reps,
                        highest_l=1,
                        adjacency_obs=None,
                        adjacency_latent=None,
                        noise_distribution = 'gamma',
                        model_class = CM,
                        methods = None,
                        location_interest = ((2, 1), )):
    """
    Parallel simulation of the experiment.
    """
    run_experiment_partial = partial(run_experiment,
                                        highest_l=highest_l,
                                        adjacency_obs=adjacency_obs,
                                        adjacency_latent=adjacency_latent,
                                        noise_distribution=noise_distribution,
                                        model_class=model_class,
                                        methods=methods,
                                        location_interest = location_interest)


    with mp.Pool(processes=mp.cpu_count()) as pool:
        print("Parallel Workers Defined")
        seeds = range(reps)
        results = pool.map(run_experiment_partial, seeds)
        print("Parallel Workers Over")

    return pd.concat(results, ignore_index=True)

def parallel_simulation_grica(reps,
                        highest_l=1,
                        adjacency_obs=None,
                        adjacency_latent=None,
                        noise_distribution = 'gamma',
                        model_class = CM):
    """
    Parallel simulation of the experiment.
    """
    run_experiment_partial = partial(run_experiment_grica,
                                        highest_l=highest_l,
                                        adjacency_obs=adjacency_obs,
                                        adjacency_latent=adjacency_latent,
                                        noise_distribution=noise_distribution,
                                        model_class=model_class)


    with tmp.Pool(processes=tmp.cpu_count()) as pool:
        print("Parallel Workers Defined")
        seeds = range(reps)
        results = pool.map(run_experiment_partial, seeds)
        print("Parallel Workers Over")

    return pd.concat(results, ignore_index=True)





def wrapper(reps,
            highest_l=1,
            adjacency_obs=None,
            adjacency_latent=None,
            noise_distribution='gamma',
            model_class = CM,
            methods = None,
            grica=True,
            location_interest = ((2, 1), )):
    """
    Wrapper for the experiment.
    """

    df = parallel_simulation(reps,
                             highest_l=highest_l,
                             adjacency_obs=adjacency_obs,
                             adjacency_latent=adjacency_latent,
                             noise_distribution=noise_distribution,
                             model_class=model_class,
                             methods=methods,
                             location_interest = location_interest)
    if grica:
        df_grica = parallel_simulation_grica(reps,
                                adjacency_obs=adjacency_obs,
                                adjacency_latent=adjacency_latent,
                                highest_l=highest_l,
                                noise_distribution=noise_distribution,
                                model_class=model_class)
        df['GRICA Estimate'] = df_grica['GRICA Estimate']
        methods += ['GRICA']

    if len(location_interest) > 1:
        for i in range(len(location_interest)):
            df[f"True Effect {i}"] = df["True Effect"].apply(lambda x: x[i])

        df = df.drop(columns=["True Effect"])

    for method in methods:
        if len(location_interest) == 1:
            df[method + ' Error'] = np.abs((df[method + ' Estimate'] - df["True Effect"])/df["True Effect"])
        else:
            df[method + ' Error'] = np.zeros(len(df))
            for i in range(len(location_interest)):
                df[method + f" Estimate {i}"] = df[method + ' Estimate'].apply(lambda x: x[i])
                df[method + f' Error {i}'] = np.abs((df[method + f" Estimate {i}"] - df[f"True Effect {i}"])/df[f"True Effect {i}"])
                df[method + ' Error'] += df[method + f' Error {i}']
            df[method + ' Error'] /= len(location_interest)
            df = df.drop(columns=[method + ' Estimate'])
    return df
