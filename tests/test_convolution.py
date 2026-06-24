import numpy as np

from pquick.convolution import _match_component_count


def test_match_component_count_promotes_scalar_beam_to_temperature_only():
    sky = np.ones((3, 10), dtype=np.complex128)
    beam = np.ones((1, 6), dtype=np.complex128)

    new_sky, new_beam = _match_component_count(sky, beam)

    assert new_sky.shape == (3, 10)
    assert new_beam.shape == (3, 6)
    # ducc0 beam slots 1-2 are spin-2 (polarised), not Q/U: the scalar beam goes
    # in the temperature slot only; copying it into 1-2 leaks E into T.
    assert np.allclose(new_beam[0], 1.0)
    assert np.allclose(new_beam[1], 0.0)
    assert np.allclose(new_beam[2], 0.0)


def test_match_component_count_rejects_incompatible_counts():
    sky = np.ones((2, 10), dtype=np.complex128)
    beam = np.ones((1, 6), dtype=np.complex128)

    try:
        _match_component_count(sky, beam)
    except ValueError as exc:
        assert "incompatible component counts" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incompatible component counts")


