import numpy as np 


def NE_convert(start: "str" , end: "str", nu: float, T= float, beam = float, value: float) -> float:
    """Convert between different noise equivalent units.

    Parameters
    ----------
    start : str
        The starting unit. Options are: 'NEI', 'NEFD', 'NET'.
    end : str
        The desired unit. Options are: 'NEI', 'NEFD', 'NET'.
    frequency : float 
        The frequency in GHz.
    value : float
        The value of the starting type to convert.

    Returns
    -------
    float
        The converted value in the above shown units
    """

    c = 299792458 # m/s
    k_B = 1.380649e-23 # J/K
    h = 6.62607015e-34 # J*s
    T_CMB = 2.725 # K`

    del_B_T = (2*h*nu**3)/c**2 * (np.exp(h*nu/(k_B*T)))/((np.exp(h*nu/(k_B*T))-1)**2) * (h*nu/(k_B*T**2))


    if start == 'NEP' and end == 'NET':
        return value / del_B_T
    elif start == 'NET' and end == 'NEP':
        return value * del_B_T
    elif start == end:
        return value
    else:
        raise ValueError("Invalid conversion types. Options are: 'NEI', 'NEFD', 'NET'.")



