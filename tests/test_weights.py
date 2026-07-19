import pytest

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


def test_pr3_hfi_weights_are_per_detector():
    # PR3 (SRoll) weights differ between a horn's two arms, unlike NPIPE.
    assert detector_map_weight("143-1a", "PR3") != detector_map_weight("143-1b", "PR3")
    assert detector_map_weight("143-1a", "PR3") > 0.0
    # NPIPE keeps them equal (per-horn).
    assert detector_map_weight("143-1a", "NPIPE") == detector_map_weight("143-1b", "NPIPE")
    # 143-8 (dead) is absent from PR3 too.
    assert has_detector_weight("143-8", "PR3") is False
    assert has_detector_weight("143-1a", "PR3") is True


def test_pr3_lfi_horn_weight_eq7():
    # Eq. 7: w = 2 / (sigma_M^2 + sigma_S^2) from Table 4; both arms share it.
    expected = 2.0 / (281.5 + 302.8)  # LFI27 M/S white-noise
    assert detector_map_weight("LFI27M", "PR3") == pytest.approx(expected)
    assert detector_map_weight("LFI27S", "PR3") == detector_map_weight("LFI27M", "PR3")


def test_pr4_is_alias_for_npipe():
    from pquick.config import _parse_data_version

    assert _parse_data_version("PR4") == "NPIPE"
    assert detector_map_weight("143-1a", "PR4") == detector_map_weight("143-1a", "NPIPE")
    assert has_detector_weight("143-1a", "PR4") is True


def test_unknown_detector_not_in_weight_set():
    # Unknown detectors are absent from both sets (so the pipeline skips them) and
    # fall back to the default weight, uniformly across modes.
    for mode in ("NPIPE", "PR3"):
        assert has_detector_weight("LFI99M", mode) is False
        assert detector_map_weight("LFI99M", mode, default=2.5) == 2.5


def test_npipe_is_per_detector_with_shared_horn_value():
    # NPIPE is a per-detector table; both arms carry the same horn weight.
    from pquick.utilities import NPIPE_DETECTOR_WEIGHTS, PR3_DETECTOR_WEIGHTS

    assert "143-1a" in NPIPE_DETECTOR_WEIGHTS and "143-1b" in NPIPE_DETECTOR_WEIGHTS
    assert NPIPE_DETECTOR_WEIGHTS["143-1a"] == NPIPE_DETECTOR_WEIGHTS["143-1b"]
    assert "143-8" not in NPIPE_DETECTOR_WEIGHTS  # dead detector absent
    # both tables cover the same detector universe
    assert set(NPIPE_DETECTOR_WEIGHTS) == set(PR3_DETECTOR_WEIGHTS)
