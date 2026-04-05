# Ethan Watanabe
# nuclear decay simulator

import numpy as np
import matplotlib.pyplot as plt
from NuDat import *

def decay_sim(N0, plot, *args):
    """
    (Docstrings generated with AI) *reviewed by Ethan Watanabe
    Simulate radioactive decay along a linear decay chain and optionally plot
    the nuclide populations over time.

    Using a simple explicit time-stepping scheme, this function computes the
    time evolution of the population of each nuclide in a decay chain defined
    by `args`, starting from an initial population in the first nuclide. It
    looks up half-lives and units via `get_nucleus`, another function coded by me, converts them to seconds,
    and assumes each nuclide decays only into the next one in the chain.

    Args:
        N0 (float): Initial population of the first nuclide at time t = 0.
        plot (bool): If True, plot population vs. time for each nuclide
            on a logarithmic y-axis.
        *args (str): One or more nuclide identifiers (e.g., symbols or names)
            that define the decay chain order from parent to final daughter.
            Each identifier must be valid input to `get_nucleus`.

    Returns:
        dict[str, numpy.ndarray]: A dictionary mapping each nuclide identifier
            to a 1D array of its population values over time.
        numpy.ndarray: A 1D array of time points (in seconds) corresponding to
            the population arrays.

    >>> populations, time = decay_sim(1e9, True, "222rn", "218po", "214pb")

    Notes:
        The maximum simulation time is set to five times the shortest half-life
        in the chain, discretized into 1000 steps, and populations are updated
        using an explicit Euler-like method.
    """
    chain = np.array([])
    half_lives = np.array([])
    nuclei = np.array([])
    for arg in args:
        nucleus = get_nucleus(arg)
        try:
            # Will execute if get_nuclei() returns a pandas series
            half_life = nucleus.loc['halflife']
            half_life = float(half_life)
            unit = nucleus.loc['halflifeUnit']
        except KeyError:
            # Will execute if get_nuclei() returns a pandas DataFrame
            half_life = nucleus.iloc[0]['halflife']
            half_life = float(half_life)
            unit = nucleus.iloc[0]['halflifeUnit']
            nucleus = nucleus.iloc[0]
        finally:
            # let's normalize to seconds
            if unit == "y":
                half_life *= 3.1536e7
            elif unit == "d":
                half_life *= 86400
            elif unit == "h":
                half_life *= 3600
            elif unit == "m":
                half_life *= 60
            elif unit == "ms":
                half_life *= 1e-3
            elif unit == "us":
                half_life *= 1e-6
            elif unit == "ns":
                half_life *= 1e-9
            elif unit == "ps":
                half_life *= 1e-12
            else:
                pass
        chain = np.append(chain, arg)
        half_lives = np.append(half_lives, half_life)
        nuclei = np.append(nuclei, nucleus)
        
    decay_constants = 0.69314718/half_lives
    tmax = 5*np.min(half_lives) # until the fastest decaying nuclei is pretty much gone
    ts = np.linspace(0, tmax, 1000)
    populations = {}
    I = 1000 # number of iterations

    for i in range(len(chain)):
        l = decay_constants[i]
        nuc_pop = np.zeros(I)
        if i == 0:
            for k in range(i, I):
                if k == 0:
                    nuc_pop[0] = N0
                else:
                    nuc_pop[k] = -l*nuc_pop[k-1]*(tmax/I) + nuc_pop[k-1]
        else:
            for k in range(i, I):
                prev_pop = populations.get(chain[i-1])
                if k == i:
                    nuc_pop[k] = prev_pop[k-1] - prev_pop[k] # add to this nucleus' population from previous nucleus' decays
                else:
                    nuc_pop[k] = (tmax/I)*(decay_constants[i-1]*prev_pop[k-1] - l*nuc_pop[k-1]) + nuc_pop[k-1]
        populations.update({chain[i]: nuc_pop})
        if plot:
            plt.plot(ts, nuc_pop, label=chain[i])
    if plot:
        plt.title("Nuclei Populations vs. Time (s)")
        plt.ylabel("Population")
        plt.xlabel("Time (s)")
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.legend()
        # plt.xscale("log")
        plt.yscale("log")
        plt.show()
    return populations, ts


if __name__ == '__main__':
    populations, time = decay_sim(1e9, True, "222rn", "218po", "214pb")
    print(f"After {time[500]:.1f} seconds we have {populations.get("218po")[500]:g}")
