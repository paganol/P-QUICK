from __future__ import annotations

import healpy as hp
import numpy as np


def accumulate_simple_iqu(
    theta: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    tod: np.ndarray,
    flags: np.ndarray,
    nside: int,
    det_weight: float,
    nest: bool = False,
) -> dict[str, np.ndarray]:
    npix = hp.nside2npix(nside)
    i_num = np.zeros(npix, dtype=np.float64)
    q_num = np.zeros(npix, dtype=np.float64)
    u_num = np.zeros(npix, dtype=np.float64)
    i_den = np.zeros(npix, dtype=np.float64)
    w_pol = np.zeros(npix, dtype=np.float64)
    hits = np.zeros(npix, dtype=np.int64)

    good = np.asarray(flags) == 0
    if not np.any(good):
        return {
            "i_num": i_num,
            "q_num": q_num,
            "u_num": u_num,
            "i_den": i_den,
            "hits": hits,
            "wpol": w_pol,
        }

    th = np.asarray(theta, dtype=np.float64)[good]
    ph = np.asarray(phi, dtype=np.float64)[good]
    ps = np.asarray(psi, dtype=np.float64)[good]
    y = np.asarray(tod, dtype=np.float64)[good]

    pix = hp.ang2pix(nside, th, ph, nest=nest)
    c2 = np.cos(2.0 * ps)
    s2 = np.sin(2.0 * ps)
    w = float(det_weight)

    np.add.at(i_num, pix, w * y)
    np.add.at(q_num, pix, w * y * c2)
    np.add.at(u_num, pix, w * y * s2)
    np.add.at(i_den, pix, w)
    np.add.at(w_pol, pix, w * (c2 * c2 + s2 * s2))
    np.add.at(hits, pix, 1)

    return {
        "i_num": i_num,
        "q_num": q_num,
        "u_num": u_num,
        "i_den": i_den,
        "hits": hits,
        "wpol": w_pol,
    }


def finalize_simple_iqu(acc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    i_num = acc["i_num"]
    q_num = acc["q_num"]
    u_num = acc["u_num"]
    i_den = acc["i_den"]
    w_pol = acc["wpol"]
    hits = acc["hits"]

    i_map = np.zeros_like(i_num)
    q_map = np.zeros_like(q_num)
    u_map = np.zeros_like(u_num)

    m_i = i_den > 0
    m_p = w_pol > 0

    i_map[m_i] = i_num[m_i] / i_den[m_i]
    q_map[m_p] = q_num[m_p] / w_pol[m_p]
    u_map[m_p] = u_num[m_p] / w_pol[m_p]

    return {
        "I": i_map,
        "Q": q_map,
        "U": u_map,
        "i_den": i_den,
        "hits": hits,
        "wpol": w_pol,
    }
