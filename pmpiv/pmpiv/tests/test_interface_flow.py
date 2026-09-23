#!/usr/bin/env python3
"""Tests for pmpiv.interface_flow."""
import numpy as np
import pandas as pd
import pytest

import pmpiv
from pmpiv import interface_flow as flow
from conftest import _make_trajectory_df


# ---------------------------------------------------------------------------
# Synthetic Lamb-Oseen vortex (analytic velocity + vorticity ground truth)
# ---------------------------------------------------------------------------

def _lamb_oseen_velocity(x, y, cx, cy, gamma, r_core):
    px, py = x - cx, y - cy
    rho = np.maximum(np.sqrt(px ** 2 + py ** 2), 1e-6)
    v_theta = (gamma / (2 * np.pi * rho)) * (1.0 - np.exp(-(rho / r_core) ** 2))
    vx = -v_theta * py / rho
    vy = v_theta * px / rho
    return vx, vy


def _lamb_oseen_peak_vorticity(gamma, r_core):
    return gamma / (np.pi * r_core ** 2)


def _make_lamb_oseen_particles(cx=0.0, cy=100.0, r_core=30.0, gamma=None,
                               x_range=(-150.0, 150.0), depth_range=(0.0, 200.0),
                               n_particles=1500, seed=0):
    if gamma is None:
        gamma = 2 * np.pi * r_core ** 2 * 50.0  # peak vorticity ~= 100 rad/s
    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_range, n_particles)
    y = rng.uniform(*depth_range, n_particles)
    vx, vy = _lamb_oseen_velocity(x, y, cx, cy, gamma, r_core)
    return pd.DataFrame({'x': x, 'depth': y, 'vx': vx, 'vy': vy}), gamma


# ---------------------------------------------------------------------------
# particle_velocity_frame / restrict_to_roi
# ---------------------------------------------------------------------------

class TestParticleVelocityFrame:
    def test_constant_velocity_recovered(self, md):
        df = _make_trajectory_df(n_particles=3, n_frames=5, dx=3.0, dy=2.0)
        result = flow.particle_velocity_frame(df, frame=0, m_metadata=md)

        # _make_trajectory_df adds +-0.05px jitter per point, so a couple percent
        # of tolerance on top of that is expected here, not a bug.
        expected_vx = 3.0 * md.FPS * md.PIXELSIZE
        expected_vy = 2.0 * md.FPS * md.PIXELSIZE
        assert result['vx'].mean() == pytest.approx(expected_vx, rel=0.05)
        assert result['vy'].mean() == pytest.approx(expected_vy, rel=0.05)

    def test_no_metadata_raises(self):
        df = _make_trajectory_df()
        with pytest.raises(FileNotFoundError):
            flow.particle_velocity_frame(df, frame=0)


class TestRestrictToRoi:
    def test_filters_x_and_frame(self):
        df = _make_trajectory_df(n_particles=4, n_frames=6, x0=0.0, dx=50.0)
        out = flow.restrict_to_roi(df, x_range=(90, 250), frame_range=(1, 3))
        assert len(out) < len(df)
        assert out['x'].between(90, 250).all()
        assert out['frame'].between(1, 3).all()


# ---------------------------------------------------------------------------
# interface_relative_velocity
# ---------------------------------------------------------------------------

class TestInterfaceRelativeVelocity:
    CX, CY, R = 0.0, -50.0, 100.0  # apex at cy + r = 50

    def test_tangent_at_apex_is_horizontal(self):
        df = pd.DataFrame({'particle': [0], 'x': [0.0], 'y': [60.0], 'vx': [2.0], 'vy': [0.0]})
        out = flow.interface_relative_velocity(df, self.CX, self.CY, self.R, x_search=(-90, 90))
        row = out.iloc[0]
        assert row['tx'] == pytest.approx(1.0, abs=1e-6)
        assert row['ty'] == pytest.approx(0.0, abs=1e-6)
        assert row['ny'] == pytest.approx(1.0, abs=1e-6)  # normal points into the liquid (+y)
        assert row['v_t'] == pytest.approx(2.0, abs=1e-6)
        assert row['v_n'] == pytest.approx(0.0, abs=1e-6)

    def test_tangent_normal_are_orthonormal_and_conserve_speed(self):
        df = pd.DataFrame({'particle': [0], 'x': [50.0], 'y': [100.0], 'vx': [1.5], 'vy': [-2.5]})
        out = flow.interface_relative_velocity(df, self.CX, self.CY, self.R, x_search=(-90, 90))
        row = out.iloc[0]
        assert row['tx'] * row['nx'] + row['ty'] * row['ny'] == pytest.approx(0.0, abs=1e-9)
        speed2 = row['vx'] ** 2 + row['vy'] ** 2
        assert row['v_t'] ** 2 + row['v_n'] ** 2 == pytest.approx(speed2, rel=1e-9)

    def test_drops_particle_above_interface(self):
        df = pd.DataFrame({'particle': [0], 'x': [0.0], 'y': [10.0], 'vx': [1.0], 'vy': [1.0]})
        out = flow.interface_relative_velocity(df, self.CX, self.CY, self.R, x_search=(-90, 90))
        assert len(out) == 0

    def test_drops_particle_outside_arc_domain(self):
        df = pd.DataFrame({'particle': [0], 'x': [150.0], 'y': [500.0], 'vx': [1.0], 'vy': [1.0]})
        out = flow.interface_relative_velocity(df, self.CX, self.CY, self.R, x_search=(-200, 200))
        assert len(out) == 0

    def test_x_search_filters_before_arc_check(self):
        df = pd.DataFrame({'particle': [0, 1], 'x': [0.0, 5.0], 'y': [60.0, 60.0],
                           'vx': [1.0, 1.0], 'vy': [1.0, 1.0]})
        out = flow.interface_relative_velocity(df, self.CX, self.CY, self.R, x_search=(-1, 1))
        assert list(out['particle']) == [0]


# ---------------------------------------------------------------------------
# velocity_field_grid + vorticity (Lamb-Oseen recovery)
# ---------------------------------------------------------------------------

class TestVelocityFieldAndVorticity:
    def test_recovers_lamb_oseen_peak_and_location(self):
        cx, cy, r_core = 0.0, 100.0, 30.0
        df, gamma = _make_lamb_oseen_particles(cx=cx, cy=cy, r_core=r_core, n_particles=2000, seed=1)

        grid = flow.velocity_field_grid(df, x_range=(-150, 150), depth_range=(0, 200), grid_step=8.0)
        omega = flow.vorticity(grid['X'], grid['Y'], grid['U'], grid['V'], grid['dx'], grid['dy'],
                               smooth_sigma=1.0)

        analytic_peak = _lamb_oseen_peak_vorticity(gamma, r_core)
        peak = np.nanmax(np.abs(omega))
        assert peak == pytest.approx(analytic_peak, rel=0.6)

        core_idx = np.unravel_index(np.nanargmax(np.abs(omega)), omega.shape)
        assert abs(grid['X'][core_idx] - cx) <= 3 * grid['dx']
        assert abs(grid['Y'][core_idx] - cy) <= 3 * grid['dy']

    def test_edges_outside_convex_hull_are_nan_not_zero(self):
        df = pd.DataFrame({'x': [0.0, 1.0, 0.5], 'depth': [0.0, 0.0, 1.0],
                           'vx': [1.0, 1.0, 1.0], 'vy': [0.0, 0.0, 0.0]})
        grid = flow.velocity_field_grid(df, x_range=(-50, 50), depth_range=(-50, 50), grid_step=10.0)
        assert np.isnan(grid['U']).any()


# ---------------------------------------------------------------------------
# detect_eddies
# ---------------------------------------------------------------------------

class TestDetectEddies:
    def _gaussian_bump(self, cx, cy, core, amp, xs, ys):
        X, Y = np.meshgrid(xs, ys)
        return X, Y, amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / core ** 2)

    def test_single_positive_eddy(self):
        xs = np.linspace(-100, 100, 41)
        ys = np.linspace(0, 200, 41)
        dx, dy = xs[1] - xs[0], ys[1] - ys[0]
        X, Y, omega = self._gaussian_bump(0.0, 100.0, 30.0, 5.0, xs, ys)

        eddies = flow.detect_eddies(omega, X, Y, dx, dy, threshold_frac=0.3)

        assert len(eddies) == 1
        e = eddies[0]
        assert e['dominant'] is True
        assert e['sign'] == 1
        assert abs(e['core_x']) <= 2 * dx
        assert abs(e['core_y'] - 100.0) <= 2 * dy
        expected_radius = 30.0 * np.sqrt(np.log(1 / 0.3))
        assert e['radius_eq'] == pytest.approx(expected_radius, rel=0.4)

    def test_counter_rotating_eddies_not_merged(self):
        xs = np.linspace(-100, 100, 61)
        ys = np.linspace(0, 100, 31)
        dx, dy = xs[1] - xs[0], ys[1] - ys[0]
        X, Y = np.meshgrid(xs, ys)
        omega = (5.0 * np.exp(-((X + 40) ** 2 + (Y - 50) ** 2) / 20.0 ** 2)
                 - 5.0 * np.exp(-((X - 40) ** 2 + (Y - 50) ** 2) / 20.0 ** 2))

        eddies = flow.detect_eddies(omega, X, Y, dx, dy, threshold_frac=0.3)

        assert len(eddies) == 2
        assert sorted(e['sign'] for e in eddies) == [-1, 1]

    def test_no_eddies_below_noise_floor(self):
        xs = np.linspace(-50, 50, 11)
        ys = np.linspace(0, 50, 11)
        X, Y = np.meshgrid(xs, ys)
        omega = np.full(X.shape, 1e-6)
        eddies = flow.detect_eddies(omega, X, Y, xs[1] - xs[0], ys[1] - ys[0])
        assert eddies == []


# ---------------------------------------------------------------------------
# penetration_depth
# ---------------------------------------------------------------------------

class TestPenetrationDepth:
    def _profile_grid(self, decay_length, n_depth=40, n_x=10, max_depth=200.0):
        depths = np.linspace(0, max_depth, n_depth)
        xs = np.linspace(0, 100, n_x)
        Y, X = np.meshgrid(depths, xs, indexing='ij')
        omega = 5.0 * np.exp(-Y / decay_length)
        return X, Y, omega

    def test_recovers_exponential_decay_depth(self):
        decay_length = 40.0
        X, Y, omega = self._profile_grid(decay_length)
        result = flow.penetration_depth(X, Y, omega, decay_frac=0.1, near_rows=2)

        expected = decay_length * np.log(10)  # depth where exp(-d/L) == 0.1
        assert result['censored'] is False
        assert result['depth'] == pytest.approx(expected, rel=0.1)

    def test_censored_when_never_decays(self):
        xs = np.linspace(0, 50, 6)
        ys = np.linspace(0, 100, 21)
        Y, X = np.meshgrid(ys, xs, indexing='ij')
        omega = np.full(Y.shape, 5.0)
        result = flow.penetration_depth(X, Y, omega, decay_frac=0.1)
        assert result['censored'] is True
        assert np.isnan(result['depth'])

    def test_mostly_nan_rows_are_skipped(self):
        X, Y, omega = self._profile_grid(decay_length=40.0, n_depth=30, n_x=10)
        # poison most of one shallow row so it should be skipped, not averaged in
        omega[2, :-1] = np.nan
        result = flow.penetration_depth(X, Y, omega, decay_frac=0.1, near_rows=2, min_valid_frac=0.5)
        assert result['censored'] is False
        assert np.isfinite(result['depth'])
