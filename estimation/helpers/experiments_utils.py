"""Wrapper for the experiment."""

from functools import partial
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pyximport
import torch.multiprocessing as tmp
from other_models.direct_lingam.simulate_data import simulate_data
from .models import CM
from .utils import reduce_list

pyximport.install(inplace=True)

@dataclass
class SimulationSettings:
    """Dataclass for simulation settings."""
    highest_l: int
    adjacency_obs: np.ndarray
    adjacency_latent: np.ndarray
    noise_distribution: str


def run_experiment(seed,
                   settings: SimulationSettings = None,
                   model_class=CM, methods=None,
                   location_interest=((2, 1),)):
    """Run an experiment with or without GRICA method."""

    print(f"Running experiment with seed {seed}\n")

    sample_sizes = [10**n for n in range(1, 8, 2)]
    n_max = max(sample_sizes)

    # Simulate data
    np.random.seed(seed)
    x, true_b = simulate_data(n_max, noise_distribution=settings.noise_distribution,
                              Lambda=settings.adjacency_obs, Gamma=settings.adjacency_latent,
                              permute_order=False)


    causal_effect_true = reduce_list([true_b[x] for x in location_interest])
    estimates_dict = {method: [] for method in methods}


    # Process different sample sizes
    for n in sample_sizes:
        x_sample = x[:n, :]
        x_sample = np.asfortranarray(x_sample - np.mean(x_sample, axis=0))

        model = model_class(x_sample, highest_l=settings.highest_l)

        if 'GRICA' in methods:
            estimates_dict['GRICA'].append(model.estimate_effect_grica())
        if 'Cumulant' in methods:
            estimates_dict['Cumulant'].append(model.estimate_effect())
        if 'Cumulant Minimization' in methods:
            estimates_dict['Cumulant Minimization'].append(model.estimate_effect_minimization())
        if 'Cross Moment' in methods:
            estimates_dict['Cross Moment'].append(model.estimate_effect_cross_moment())
        if 'Relvlingam' in methods:
            _, b_estimate = model.fit(x_sample)
            estimates_dict['Relvlingam'].append(reduce_list([b_estimate[x] for x in location_interest]))
        if 'Min Norm' in methods:
            estimates_dict['Min Norm'].append(model.estimate_effect_min_norm())

    # Create DataFrame
    seed_df = pd.DataFrame({'Seed': [seed] * len(sample_sizes), 'Sample Size': sample_sizes})

    seed_df['True Effect'] = [causal_effect_true] * len(sample_sizes)
    for method in methods:
        seed_df[method + ' Estimate'] = estimates_dict[method]

    return seed_df

def parallel_simulation(reps,
                        settings: SimulationSettings,
                        model_class=CM, methods=None,
                        location_interest=((2, 1),)):
    """
    Run the experiments in parallel.
    """
    run_experiment_partial = partial(run_experiment,
                                     settings = settings,
                                     model_class=model_class, methods=methods,
                                     location_interest=location_interest)

    context = tmp.get_context('spawn')

    with context.Pool(processes=context.cpu_count()) as pool:
        print("Running experiments in parallel...")
        seeds = range(reps)
        results = pool.map(run_experiment_partial, seeds)
        print("Parallel execution completed.")

    return pd.concat(results, ignore_index=True)

def wrapper(reps,
            settings: SimulationSettings,
            model_class=CM, methods=None,
            location_interest=((2, 1),)):
    """
    Wrapper for the experiment.
    """
    df = parallel_simulation(reps,
                             settings=settings,
                             model_class=model_class,
                             methods=methods,
                             location_interest=location_interest)

    # Handle multiple locations
    if len(location_interest) > 1:
        for i in range(len(location_interest)):
            df[f"True Effect {i}"] = df["True Effect"].apply(lambda x: x[i])
        df.drop(columns=["True Effect"], inplace=True)

    # Compute errors
    for method in methods:
        if len(location_interest) == 1:
            df[method + ' Error'] = df[method + ' Estimate'] - df["True Effect"]
            df[method + ' Error'] = np.abs(df[method + ' Error'] / df["True Effect"])
        else:
            df[method + ' Error'] = np.zeros(len(df))
            for i in range(len(location_interest)):
                df[method + f" Estimate {i}"] = df[method + ' Estimate'].apply(lambda x: x[i])
                df[method + f' Error {i}'] = df[method + f" Estimate {i}"] - df[f"True Effect {i}"]
                df[method + f' Error {i}'] = np.abs(df[method + f' Error {i}'] / df[f"True Effect {i}"])
                df[method + ' Error'] += df[method + f' Error {i}']
            df[method + ' Error'] /= len(location_interest)
            df.drop(columns=[method + ' Estimate'], inplace=True)

    return df
