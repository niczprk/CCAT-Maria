import numpy as np

# Physical constants (SI)
C = 299792458.0                 # m/s
K_B = 1.380649e-23              # J/K
H = 6.62607015e-34              # J*s
T_CMB = 2.7255                  # K
JY = 1e-26                       # W m^-2 Hz^-1


def dBdT_planck(nu_hz: float, T: float = T_CMB) -> float:
    """
    Planck-law derivative dB_nu/dT evaluated at temperature T.

    Returns
    -------
    dB/dT : float
        Units: W m^-2 Hz^-1 sr^-1 K^-1
    """
    x = H * nu_hz / (K_B * T)
    ex = np.exp(x)
    pref = 2.0 * H * nu_hz**3 / C**2
    # d/dT[1/(e^x-1)] = e^x/(e^x-1)^2 * (x/T)
    return pref * (ex / (ex - 1.0)**2) * (x / T)


def beam_solid_angle_gaussian(fwhm_arcsec: float) -> float:
    """
    Gaussian beam solid angle Omega_beam ≈ 1.133 * theta_FWHM^2.

    Parameters
    ----------
    fwhm_arcsec : float
        Beam FWHM in arcseconds.

    Returns
    -------
    Omega_beam : float
        Beam solid angle in steradians.
    """
    theta_rad = fwhm_arcsec * (np.pi / (180.0 * 3600.0))
    return 1.133 * theta_rad**2


def active_beams(Ndet: float, yield_frac: float = 0.8, pol_mode: str = "broadband") -> float:
    """
    Compute number of active beams on sky based on detector count and polarization scheme.

    pol_mode:
      - "broadband": 1 detector per polarization per beam => Nbeams = Ndet/2
      - "eor_spec":  both pols into one KID => Nbeams = Ndet

    Returns
    -------
    Nbeams_active : float
    """
    pol_mode = pol_mode.lower()
    if pol_mode in ("broadband", "bb"):
        Nbeams = Ndet / 2.0
    elif pol_mode in ("eor_spec", "eorspec", "spec"):
        Nbeams = Ndet
    else:
        raise ValueError("pol_mode must be 'broadband' or 'eor_spec'.")
    return yield_frac * Nbeams


def convert_noise_equivalent(
    start: str,
    end: str,
    nu_GHz: float,
    value: float,
    beam_arcsec: float | None = None,
    Ndet: float | None = None,
    yield_frac: float = 0.8,
    pol_mode: str = "broadband",
    T: float = T_CMB,
    matched_filter: bool = False,
) -> float:
    """
    Convert between NEI, NET, and NEFD using the conventions in your table notes.

    Definitions expected:
      - NEI:  Jy sr^-1 sqrt(s)   (surface brightness noise)
      - NET:  K sqrt(s)          (thermodynamic CMB temperature noise)
      - NEFD: Jy sqrt(s) beam^-1 (point-source flux density noise per beam)

    Parameters
    ----------
    start, end : str
        'NEI', 'NET', 'NEFD' (case-insensitive)
    nu_GHz : float
        Frequency in GHz.
    value : float
        Value in the units of `start`.
    beam_arcsec : float, optional
        Beam FWHM in arcsec. Required for any conversion involving NEFD.
    Ndet : float, optional
        Total number of detectors. Required for NEI <-> NEFD if your NEI is *array-combined*
        (as in the top table notes).
    yield_frac : float
        Active detector yield (default 0.8).
    pol_mode : str
        'broadband' (Nbeams=Ndet/2) or 'eor_spec' (Nbeams=Ndet).
    T : float
        Temperature for dB/dT (default T_CMB).
    matched_filter : bool
        If True, use Omega_eff = 2*Omega_beam (Gaussian matched-filter convention).
        If False, use Omega_eff = Omega_beam ("Jy per beam" convention).

    Returns
    -------
    float
        Converted value in units of `end`.
    """
    start = start.upper()
    end = end.upper()

    if start == end:
        return value

    nu_hz = nu_GHz * 1e9
    dBdT = dBdT_planck(nu_hz, T=T)  # SI: W m^-2 Hz^-1 sr^-1 K^-1

    def nei_to_net(nei_jy_sr: float) -> float:
        return (nei_jy_sr * JY) / dBdT  # K sqrt(s)

    def net_to_nei(net_k: float) -> float:
        return (net_k * dBdT) / JY  # Jy sr^-1 sqrt(s)

    if (start == "NEFD") or (end == "NEFD"):
        if beam_arcsec is None:
            raise ValueError("beam_arcsec is required for conversions involving NEFD.")

        Omega_beam = beam_solid_angle_gaussian(beam_arcsec)
        Omega_eff = (2.0 * Omega_beam) if matched_filter else Omega_beam

        if Ndet is None and (start == "NEI" or end == "NEI"):
            raise ValueError("Ndet is required for NEI <-> NEFD with array-combined NEI (top table).")

        Nbeams_active = None
        if Ndet is not None:
            Nbeams_active = active_beams(Ndet, yield_frac=yield_frac, pol_mode=pol_mode)
            sqrtN = np.sqrt(Nbeams_active)

    if start == "NEI" and end == "NET":
        return nei_to_net(value)

    if start == "NET" and end == "NEI":
        return net_to_nei(value)

    if start == "NEI" and end == "NEFD":
        return value * Omega_eff * sqrtN

    if start == "NEFD" and end == "NEI":
        return value / (Omega_eff * sqrtN)

    if start == "NET" and end == "NEFD":
        # NET -> NEI -> NEFD
        nei = net_to_nei(value)
        return nei * Omega_eff * sqrtN

    if start == "NEFD" and end == "NET":
        # NEFD -> NEI -> NET
        nei = value / (Omega_eff * sqrtN)
        return nei_to_net(nei)

    raise ValueError("Invalid conversion types. Options: 'NEI', 'NET', 'NEFD'.")

# Example usage:

if __name__ == "__main__":


    nefd = convert_noise_equivalent(
    start="NEI", end="NEFD",
    nu_GHz=850,
    beam_arcsec=15,
    Ndet=20808,
    value=479000)

    net = convert_noise_equivalent(
        start="NEI", end="NET",
        nu_GHz=850,
        value=479000,  # Jy sqrt(s)
)

print(nefd)         # Jy sqrt(s)
print(nefd * 1e3)   # mJy sqrt(s)

print(net)  # K sqrt(s)
print(net * 1e6)  # uK sqrt(s)

