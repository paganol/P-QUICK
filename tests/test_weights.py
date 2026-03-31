from pquick.weights import detector_map_weight


def test_qp_planck_weight_lookup_hfi_psb_arm():
    assert detector_map_weight("100-1a") == detector_map_weight("100-1b")
    assert detector_map_weight("100-1a") > 0.0


def test_qp_planck_weight_lookup_lfi_arm():
    assert detector_map_weight("LFI27M") == detector_map_weight("LFI27S")
    assert detector_map_weight("LFI27M") > 0.0


def test_weight_fallback_default():
    assert detector_map_weight("UNKNOWN", default=2.5) == 2.5
