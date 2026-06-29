from pquick.utilities import detector_map_weight, has_detector_weight, is_psb


def test_is_psb_matches_qp_planck_flag():
    # qp_planck: psb = det[-1] in "abMS"; SWBs (143-5..8, 545/857) are not.
    for det in ("143-1a", "143-1b", "100-2a", "LFI27M", "LFI27S"):
        assert is_psb(det) is True
    for det in ("143-5", "143-8", "545-1", "857-2"):
        assert is_psb(det) is False


def test_non_working_detectors_have_no_weight():
    # The weight table is the good-detector list; 143-8 and 545-3 (RTS noise) are out.
    for det in ("143-1a", "143-5", "545-1", "545-4"):
        assert has_detector_weight(det) is True
    for det in ("143-8", "545-3"):
        assert has_detector_weight(det) is False


def test_qp_planck_weight_lookup_hfi_psb_arm():
    assert detector_map_weight("100-1a") == detector_map_weight("100-1b")
    assert detector_map_weight("100-1a") > 0.0


def test_qp_planck_weight_lookup_lfi_arm():
    assert detector_map_weight("LFI27M") == detector_map_weight("LFI27S")
    assert detector_map_weight("LFI27M") > 0.0


def test_weight_fallback_default():
    assert detector_map_weight("UNKNOWN", default=2.5) == 2.5
