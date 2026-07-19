from pquick.io import load_rimo_detectors


def test_npipe_focal_plane_table():
    det = load_rimo_detectors("NPIPE")
    assert len(det) == 74
    # HFI reference detector: orientation in psi_uv (npipe/R4.00 convention).
    d = det["100-1a"]
    assert abs(d["psi_uv"] - 23.100926005468104) < 1e-12
    assert abs(d["psi_pol"] - 0.21718725535964634) < 1e-12
    assert 0 < d["epsilon"] < 0.1 and 0.9 < d["rho_pol"] < 1.0
    # LFI M arm carries the 90 deg polarization rotation in psi_pol.
    assert abs(det["LFI18M"]["psi_pol"] - 90.2) < 1e-12
    assert det["LFI18S"]["psi_pol"] == 0.0


def test_pr3_focal_plane_table():
    det = load_rimo_detectors("PR3")
    assert len(det) == 74
    d = det["100-1a"]
    # HFI R2.00 orientation (stored there as PSI_POL) mapped into psi_uv;
    # PR3 has no separate pol-axis fine offset.
    assert abs(d["psi_uv"] - 23.1009) < 1e-3
    assert d["psi_pol"] == 0.0
    # PR3 epsilon differs from NPIPE's.
    assert abs(d["epsilon"] - 0.0272) < 1e-12
    npipe = load_rimo_detectors("NPIPE")["100-1a"]
    assert d["epsilon"] != npipe["epsilon"]
    # LFI R2.50: same convention as npipe, small nonzero epsilon.
    assert abs(det["LFI18M"]["psi_pol"] - 90.2) < 1e-12
    assert det["LFI18M"]["epsilon"] > 0.0


def test_pr4_alias_and_default():
    assert load_rimo_detectors().keys() == load_rimo_detectors("PR4").keys()
    assert load_rimo_detectors("pr4")["100-1a"]["psi_uv"] == load_rimo_detectors("NPIPE")["100-1a"]["psi_uv"]
