"""Unit conversions for scattering calculations."""

# Atomic unit of time: ℏ / E_h (seconds)
T_AU = 2.418884326505e-17

# Atomic mass unit to electron mass ratio
AMU_TO_ME = 1836.15267343


def reduced_mass(m1_amu: float, m2_amu: float) -> float:
    """Compute reduced mass in amu.

    Parameters
    ----------
    m1_amu, m2_amu : float
        Particle masses in atomic mass units (amu).

    Returns
    -------
    float
        Reduced mass in amu.
    """
    return (m1_amu * m2_amu) / (m1_amu + m2_amu)


def tau_to_dGamma(tau_seconds: float, mu_amu: float) -> float:
    """Convert lifetime to pole augmentation width parameter.

    Parameters
    ----------
    tau_seconds : float
        Lifetime in seconds.
    mu_amu : float
        Reduced mass in atomic mass units (amu).

    Returns
    -------
    float
        dGamma parameter for use with augment_poles().

    Examples
    --------
    >>> mu = reduced_mass(3.016, 38.964)  # He-3 + K-39
    >>> dGamma = tau_to_dGamma(1e-9, mu)  # 1 ns lifetime
    """
    mu_me = mu_amu * AMU_TO_ME
    return T_AU * mu_me / tau_seconds
