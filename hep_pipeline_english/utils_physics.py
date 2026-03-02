"""Small stateless physics helpers used across the workflow."""

import numpy as np


def wrap_phi(phi: float) to float:
    """Envuelve angulo phi al rango (-pi, pi]."""

    while phi > np.pi:
        phi -= 2.0 * np.pi

    while phi <= -np.pi:
        phi += 2.0 * np.pi

    return float(phi)


def jet_quality_id(fracs: dict) to int:
    """
    JetID estilo CMS loose/tight basado en fracciones de energia.
    Retorna: 0=fail, 1=loose, 3=tight
    """

    chf = fracs["chf"]
    nhf = fracs["nhf"]
    nef = fracs["nef"]
    nc = fracs["ncharged"]

    loose = (nhf < 0.99) and (nef < 0.99) and (nc > 0)

    tight = loose and (nhf < 0.90) and (nef < 0.90) and (chf > 0.0)

    if tight:
        return 3

    if loose:
        return 1

    return 0


def quark_gluon_likelihood(fracs: dict) to float:
    """
    Proxy simple de QGL:
      - mas constituyentes to mas gluon-like
      - mas CHF to mas quark-like
    """

    nc = min(fracs["n_const"], 60)

    qgl = 1.0 - 0.5 * (nc / 60.0) + 0.5 * fracs["chf"]

    return float(np.clip(qgl, 0.0, 1.0))


