import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "nndc_nudat_data_export.csv"

df = pd.read_csv(DATA_PATH)
ground_states_only = df[df["levelEnergy(MeV)"] == "0"]
nuclei_names = ground_states_only["name"]

df["name"] = df["name"].str.lower()
df.index = df["name"]
df = df.drop(columns=["name"])


class NucleusNotFoundError(Exception):
    pass

class AmbiguousNucleusError(Exception):
    pass

def q_value(reactants, products):
    """
    Will take the reactants and products of a nuclear reaction and provide the Q-value in keV
    Calculates to an accuracy of three sig figs
    :param reactants: list/tuple of the reactant nuclei as strings
    :param products: list/tuple of the product nuclei as strings
    :return: Q-value of the reaction
    """
    Q = 0
    for reactant in reactants:
        nucleus = get_nucleus(reactant)
        try:
            # Will execute if get_nuclei() returns a pandas series
            mass_excess = nucleus.loc['massExcess(keV)']
        except KeyError:
            # Will execute if get_nuclei() returns a pandas DataFrame
            mass_excess = nucleus.iloc[0]['massExcess(keV)']
        finally:
            Q += mass_excess

    for product in products:
        nucleus = get_nucleus(product)
        try:
            # Will execute if get_nuclei() returns a pandas series
            mass_excess = nucleus.loc['massExcess(keV)']
        except KeyError:
            # Will execute if get_nuclei() returns a pandas DataFrame
            mass_excess = nucleus.iloc[0]['massExcess(keV)']
        finally:
            Q -= mass_excess

    return float(Q)

def neutron_sep(nucleus_str):
    """
    Calculates the neutron separation energy from a given nucleus.
    :param nucleus_str: String of the nucleus we want to calculate the neutron separation energy for
    :return: Float of the neutron separation energy in keV
    """
    # We start by subtracting the mass excess of the given nucleus as per the formula
    Sn = 8071.32 # start with mass excess of a neutron in keV
    nucleus = get_nucleus(nucleus_str)
    try:
        # Will execute if get_nuclei() returns a pandas series
        mass_excess = nucleus.loc['massExcess(keV)']
    except KeyError:
        # Will execute if get_nuclei() returns a pandas DataFrame
        mass_excess = nucleus.iloc[0]['massExcess(keV)']
    finally:
        Sn -= mass_excess

    A = ""
    element = ""
    for char in nucleus_str: # will split the nuclei string into the mass number and the element name
        if char.isdigit():
            A += char
        else:
            element += char

    A = int(A)
    lessNeutron_nucleus = get_nucleus(f"{A - 1}{element}")
    try:
        # Will execute if get_nuclei() returns a pandas series
        mass_excess = lessNeutron_nucleus.loc['massExcess(keV)']
    except KeyError:
        # Will execute if get_nuclei() returns a pandas DataFrame
        mass_excess = lessNeutron_nucleus.iloc[0]['massExcess(keV)']
    finally:
        Sn += mass_excess

    return Sn

def proton_sep(nucleus_str):
    """
    Calculates the proton separation energy from a given nucleus.
    :param nucleus_str: String of the nucleus we want to calculate the proton separation energy for
    :return: Float of the proton separation energy in keV
    """

    Sp = 7288.97 # start with mass excess of a proton in keV
    nucleus = get_nucleus(nucleus_str)
    try:
        # Will execute if get_nuclei() returns a pandas series
        mass_excess = nucleus.loc['massExcess(keV)']
    except KeyError:
        # Will execute if get_nuclei() returns a pandas DataFrame
        mass_excess = nucleus.iloc[0]['massExcess(keV)']
    finally:
        Sp -= mass_excess

    A = ""
    element = ""
    for char in nucleus_str:  # will split the nuclei string into the mass number and the element name
        if char.isdigit():
            A += char
        else:
            element += char
    A = int(A)
    Z = nucleus.loc["z"]
    new_element = get_atom(Z-1)

    lessProton_nucleus = get_nucleus(f"{A - 1}{new_element}")
    print(lessProton_nucleus)
    try:
        # Will execute if get_nuclei() returns a pandas series
        mass_excess = lessProton_nucleus.loc['massExcess(keV)']
    except KeyError:
        # Will execute if get_nuclei() returns a pandas DataFrame
        mass_excess = lessProton_nucleus.iloc[0]['massExcess(keV)']
    finally:
        Sp += mass_excess

    return Sp

def alpha_sep(nucleus_str):
    """
    Calculates the alpha separation energy from a given nucleus.
    :param nucleus_str: String of the nucleus we want to calculate the alpha separation energy for
    :return: Float of the alpha separation energy in keV
    """

    Sa = 2424.92 # start with mass excess of an alpha particle in keV
    nucleus = get_nucleus(nucleus_str)
    try:
        # Will execute if get_nuclei() returns a pandas series
        mass_excess = nucleus.loc['massExcess(keV)']
    except KeyError:
        # Will execute if get_nuclei() returns a pandas DataFrame
        mass_excess = nucleus.iloc[0]['massExcess(keV)']
    finally:
        Sa -= mass_excess

    A = ""
    element = ""
    for char in nucleus_str:  # will split the nuclei string into the mass number and the element name
        if char.isdigit():
            A += char
        else:
            element += char
    A = int(A)
    Z = nucleus.loc["z"]
    new_element = get_atom(Z-2)

    lessAlpha_nucleus = get_nucleus(f"{A - 4}{new_element}")
    try:
        # Will execute if get_nuclei() returns a pandas series
        mass_excess = lessAlpha_nucleus.loc['massExcess(keV)']
    except KeyError:
        # Will execute if get_nuclei() returns a pandas DataFrame
        mass_excess = lessAlpha_nucleus.iloc[0]['massExcess(keV)']
    finally:
        Sa += mass_excess

    return Sa

def get_atom(z_value):
    zs = df.loc[:, "z"]
    nuclei = ""
    if 0 <= z_value <= 118:
        pass
    else:
        raise NucleusNotFoundError

    for label, value in zs.items():
        if value == z_value:
            nuclei = label
            break
    atom_name = ""
    for char in nuclei: # will split the nuclei string into the mass number and the element name
        if char.isdigit():
            pass
        else:
            atom_name += char
    return atom_name

def get_nucleus(nucleus):
    """
    Will return a pandas series/dataframe of the data of a certain nucleus
    :param nucleus: a string of the nucleus in AX form
    :return: A pandas series/dataframe of the data of the nucleus
    """
    try:
        nucleus = nucleus.lower()
    except AttributeError:
        print("Please enter a valid nucleus")
        raise NucleusNotFoundError

    try:
        return df.loc[nucleus]
    except KeyError:
        print("Nucleus not found")
        raise NucleusNotFoundError

def SEMF(nucleus_str):
    """
    Returns the binding energy approximation using the Semi-Empircal Mass Formula in MeV.
    :param nucleus_str: String of the nucleus we want to calculate the binding energy for
    :return: Binding energy of the nucleus in MeV
    """
    nucleus = get_nucleus(nucleus_str)
    a_v = 15.5
    a_s = 16.8
    a_c = 0.72
    a_sym = 23

    try:
        # Will execute if get_nuclei() returns a pandas series
        n = nucleus.loc['n']
        z = nucleus.loc['z']
    except KeyError:
        # Will execute if get_nuclei() returns a pandas DataFrame
        n = nucleus.iloc[0]['n']
        z = nucleus.iloc[0]['z']
    A = n + z
    a_p = 0
    if n % 2 == 0 and z % 2 == 0:
        a_p = 34
    elif n % 2 != 0 and z % 2 != 0:
        a_p = -34
    else:
        pass

    SEMF = a_v * A - a_s * A**(2/3) - a_c * z*(z-1)/(A**(1/3)) - a_sym * (A - 2*z)**2 / A + a_p * A**(-0.75)
    return SEMF


def BindingEnergyCurve():
    ns = np.array([])
    zs = np.array([])
    for label, value in df.loc[:, "n"].items():
        ns = np.append(ns, value)
    for label, value in df.loc[:, "z"].items():
        zs = np.append(zs, value)
    # print(ns)
    # print(ns.size)
    As = ns + zs
    # print(np.size(As))

    binding_energies = np.array([])
    binding_per_nucleon = np.array([])
    i = 0
    for nucleus in df.index:
        binding_energies = np.append(binding_energies, SEMF(nucleus))
        binding_per_nucleon = np.append(binding_per_nucleon, binding_energies[i] / As[i])
        i += 1

    # print(binding_energies[:10])
    # print(binding_energies.size)
    # print(binding_per_nucleon[:10])
    # print(binding_per_nucleon.size)
    plt.scatter(As, binding_per_nucleon)
    plt.xlabel("Mass Number A")
    plt.ylabel("Binding Energy Per Nucleon (MeV)")
    plt.title("Binding Energy Per Nucleon vs. Mass Number")

    plt.show()



def main():
    # print(alpha_sep("12C"))
    # BindingEnergyCurve()
    # print(SEMF("40k"))
    # print(df[df["levelEnergy(MeV)"] == "0"])
    pass


if __name__ == '__main__':
    main()
