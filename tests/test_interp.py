"""Tests for healjax.interp."""

import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import healjax
from healjax.coord import eq2top_m, rot_m, thphi2xyz
from healjax.interp import interpolate_map, rotate_interpolate_and_sum

healpy = pytest.importorskip("healpy")

NSIDE = 16
NPIX = 12 * NSIDE ** 2

# healjax computes interpolation weights in FLOAT_TYPE (float32), so agreement
# with healpy's double-precision result is limited to ~1e-6 relative.
RTOL = 1e-5
ATOL = 1e-6


def ring_pix_angles(nside):
    return healpy.pix2ang(nside, np.arange(12 * nside ** 2), nest=False)


def smooth_map(nside, seed=0):
    """A band-limited (hence smoothly interpolable) test map."""
    th, phi = ring_pix_angles(nside)
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(size=3)
    return (1.0 + a * np.cos(th) + b * np.sin(th) * np.cos(phi)
            + c * np.sin(th) ** 2 * np.sin(2 * phi))


class TestInterpolateMap:

    def test_reproduces_pixel_centres(self):
        """At a pixel centre the four weights collapse onto that pixel."""
        m = jnp.asarray(smooth_map(NSIDE))
        th, phi = ring_pix_angles(NSIDE)
        out = interpolate_map(NSIDE, m, th, phi)
        np.testing.assert_allclose(np.asarray(out), np.asarray(m),
                                   rtol=RTOL, atol=ATOL)

    def test_matches_healpy_get_interp_val(self):
        m = smooth_map(NSIDE, seed=1)
        rng = np.random.default_rng(2)
        th = np.arccos(rng.uniform(-1, 1, 200))
        phi = rng.uniform(0, 2 * np.pi, 200)
        ours = np.asarray(interpolate_map(NSIDE, jnp.asarray(m), th, phi))
        theirs = healpy.get_interp_val(m, th, phi, nest=False)
        np.testing.assert_allclose(ours, theirs, rtol=RTOL, atol=ATOL)

    def test_xyz_and_thphi_paths_agree(self):
        m = jnp.asarray(smooth_map(NSIDE, seed=3))
        rng = np.random.default_rng(4)
        th = np.arccos(rng.uniform(-1, 1, 50))
        phi = rng.uniform(0, 2 * np.pi, 50)
        x, y, z = thphi2xyz(th, phi)
        a = interpolate_map(NSIDE, m, th, phi)
        b = interpolate_map(NSIDE, m, x, y, z)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=RTOL, atol=ATOL)

    def test_constant_map_is_preserved(self):
        """Interpolation weights sum to 1, so a constant map stays constant."""
        m = jnp.full(NPIX, 3.25)
        rng = np.random.default_rng(5)
        th = np.arccos(rng.uniform(-1, 1, 100))
        phi = rng.uniform(0, 2 * np.pi, 100)
        out = interpolate_map(NSIDE, m, th, phi)
        np.testing.assert_allclose(np.asarray(out), 3.25, rtol=RTOL)

    def test_multifrequency_map(self):
        """Trailing axes are carried through independently."""
        base = smooth_map(NSIDE, seed=6)
        m = jnp.asarray(np.stack([base, 2 * base, -base], axis=-1))
        rng = np.random.default_rng(7)
        th = np.arccos(rng.uniform(-1, 1, 30))
        phi = rng.uniform(0, 2 * np.pi, 30)
        out = np.asarray(interpolate_map(NSIDE, m, th, phi))
        assert out.shape == (30, 3)
        np.testing.assert_allclose(out[:, 1], 2 * out[:, 0], rtol=RTOL)
        np.testing.assert_allclose(out[:, 2], -out[:, 0], rtol=RTOL)

    def test_scalar_coordinates(self):
        m = jnp.asarray(smooth_map(NSIDE, seed=8))
        out = interpolate_map(NSIDE, m, 1.0, 2.0)
        assert np.ndim(out) == 0
        np.testing.assert_allclose(
            float(out), healpy.get_interp_val(np.asarray(m), 1.0, 2.0),
            rtol=RTOL,
        )

    def test_is_differentiable(self):
        """The whole point of the JAX path: gradients flow through."""
        m = jnp.asarray(smooth_map(NSIDE, seed=9))

        def f(th):
            return interpolate_map(NSIDE, m, th, 0.7)

        g = jax.grad(f)(1.0)
        assert np.isfinite(g)
        # finite-difference check
        eps = 1e-5
        fd = (f(1.0 + eps) - f(1.0 - eps)) / (2 * eps)
        np.testing.assert_allclose(float(g), float(fd), rtol=1e-3)

    def test_python_scalar_keeps_float64(self):
        """Weakly typed scalar coordinates must not demote to float32."""
        m = jnp.asarray(smooth_map(NSIDE, seed=11))
        scalar = interpolate_map(NSIDE, m, 1.0 + 1e-6, 0.7)
        array = interpolate_map(NSIDE, m, jnp.float64(1.0 + 1e-6), 0.7)
        np.testing.assert_allclose(float(scalar), float(array),
                                   rtol=0, atol=1e-14)

    def test_nside_is_static_under_jit(self):
        """A traced nside must be rejected, documenting the static_argnums."""
        m = jnp.asarray(smooth_map(NSIDE, seed=10))
        with pytest.raises(Exception):
            jax.jit(interpolate_map)(jnp.int32(NSIDE), m, 1.0, 2.0)


class TestRotateInterpolateAndSum:

    def setup_method(self):
        self.beam = np.abs(smooth_map(NSIDE, seed=20)) + 0.5
        self.nfreq = 4
        self.beam_map = jnp.asarray(
            np.stack([self.beam * (1 + 0.1 * i) for i in range(self.nfreq)],
                     axis=-1)
        )
        th, phi = ring_pix_angles(NSIDE)
        self.crds = jnp.asarray(np.asarray(thphi2xyz(th, phi)))
        self.sky = jnp.asarray(
            np.repeat(smooth_map(NSIDE, seed=21)[:, None], self.nfreq, axis=-1)
        )

    def test_shape(self):
        rots = jnp.asarray(np.stack([rot_m(a, np.array([0.0, 0.0, 1.0]))
                                     for a in np.linspace(0, 1, 5)]))
        out = rotate_interpolate_and_sum(NSIDE, self.beam_map, self.sky,
                                         self.crds, rots)
        assert out.shape == (5, self.nfreq)
        assert np.all(np.isfinite(np.asarray(out)))

    def test_identity_rotation_matches_direct_weighted_mean(self):
        eye = jnp.asarray(np.eye(3)[None])
        out = np.asarray(rotate_interpolate_and_sum(
            NSIDE, self.beam_map, self.sky, self.crds, eye))
        wgt = np.asarray(self.beam_map)
        expected = ((wgt * np.asarray(self.sky)).sum(axis=0)
                    / wgt.sum(axis=0))
        np.testing.assert_allclose(out[0], expected, rtol=RTOL)

    def test_uniform_sky_gives_that_value(self):
        """A constant sky integrates to its own value for any rotation."""
        sky = jnp.full((NPIX, self.nfreq), 7.0)
        rots = jnp.asarray(np.stack([rot_m(a, np.array([0.0, 1.0, 0.0]))
                                     for a in (0.0, 0.3, 1.1)]))
        out = np.asarray(rotate_interpolate_and_sum(
            NSIDE, self.beam_map, sky, self.crds, rots))
        np.testing.assert_allclose(out, 7.0, rtol=RTOL)

    def test_rotation_actually_changes_the_answer(self):
        rots = jnp.asarray(np.stack([
            np.eye(3), rot_m(1.0, np.array([0.0, 0.0, 1.0]))]))
        out = np.asarray(rotate_interpolate_and_sum(
            NSIDE, self.beam_map, self.sky, self.crds, rots))
        assert not np.allclose(out[0], out[1])

    def test_accepts_eq2top_m_stack(self):
        ha = np.linspace(-0.3, 0.3, 6)
        rots = jnp.asarray(np.asarray(eq2top_m(ha, np.full(6, 0.6))))
        out = np.asarray(rotate_interpolate_and_sum(
            NSIDE, self.beam_map, self.sky, self.crds, rots))
        assert out.shape == (6, self.nfreq)
        assert np.all(np.isfinite(out))

    def test_single_frequency_map(self):
        beam = jnp.asarray(self.beam)
        sky = jnp.asarray(smooth_map(NSIDE, seed=21))
        rots = jnp.asarray(np.eye(3)[None])
        out = np.asarray(rotate_interpolate_and_sum(
            NSIDE, beam, sky, self.crds, rots))
        assert out.shape == (1,)


class TestTopLevelReexports:

    def test_symbols_available_from_package_root(self):
        assert healjax.interpolate_map is interpolate_map
        assert healjax.rotate_interpolate_and_sum is rotate_interpolate_and_sum
