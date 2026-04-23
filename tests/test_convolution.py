import numpy as np

from pquick.convolution import _match_component_count, convolve_timeline


def test_match_component_count_promotes_scalar_beam_to_temperature_only():
    sky = np.ones((3, 10), dtype=np.complex128)
    beam = np.ones((1, 6), dtype=np.complex128)

    new_sky, new_beam = _match_component_count(sky, beam)

    assert new_sky.shape == (3, 10)
    assert new_beam.shape == (3, 6)
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


def test_convolve_timeline_reuses_interpolator_from_cache_for_same_npoints():
    sky = np.ones((1, 4), dtype=np.complex128)
    beam = np.ones((1, 4), dtype=np.complex128)
    ptg = np.zeros((5, 3), dtype=np.float64)

    created = 0

    class _FakeInterpolator:
        def interpol(self, ptg_thetaphipsi):
            n = np.asarray(ptg_thetaphipsi).shape[0]
            return np.ones(n, dtype=np.float64)

    def _factory(**kwargs):
        nonlocal created
        created += 1
        assert kwargs["npoints"] == 5
        return _FakeInterpolator()

    cache: dict[int, object] = {}
    out1 = convolve_timeline(
        sky_alm=sky,
        beam_alm=beam,
        ptg_thetaphipsi=ptg,
        lmax=2,
        mmax=1,
        interpolator_cache=cache,
        interpolator_factory=_factory,
    )
    out2 = convolve_timeline(
        sky_alm=sky,
        beam_alm=beam,
        ptg_thetaphipsi=ptg,
        lmax=2,
        mmax=1,
        interpolator_cache=cache,
        interpolator_factory=_factory,
    )

    assert created == 1
    assert 5 in cache
    assert out1.shape == (5,)
    assert out2.shape == (5,)