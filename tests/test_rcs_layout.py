from __future__ import annotations

import ast
import math
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_SOURCE = (
    ROOT
    / "source/lunar_rocket_lab/lunar_rocket_lab/tasks/direct/lunar_lander/lunar_lander_env_cfg.py"
)


def _extract_literal(source: str, name: str) -> object:
    """Pull `name = <literal>` out of the cfg source and literal_eval it.

    Mirrors the regex-extraction style already used by
    tests/test_terrain_landability.py's _foot_offsets_xy -- these tests must
    stay import-free (no isaaclab/torch) per the static-suite convention, so
    cfg constants are parsed as text, not imported.
    """
    match = re.search(rf"^\s*{name}\s*=\s*(.+?)\s*$", source, re.MULTILINE)
    if match is None:
        raise AssertionError(f"could not find `{name} = <value>` in cfg source")
    return ast.literal_eval(match.group(1))


def _extract_tuple_block(source: str, name: str) -> tuple:
    match = re.search(rf"{name}\s*=\s*\((.*?)\n    \)", source, re.DOTALL)
    if match is None:
        raise AssertionError(f"could not find `{name} = (...)` block in cfg source")
    return ast.literal_eval("(" + match.group(1) + "\n)")


def _cross(a: tuple, b: tuple) -> tuple:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: tuple) -> float:
    return math.sqrt(sum(x * x for x in v))


class RcsLayoutTests(unittest.TestCase):
    """Verify the RCS thruster geometry/authority math without needing a GPU.

    This exists because the TVC->RCS conversion shipped an actuator-authority
    bug (max_rcs_thrust_n=1.2 N gave ~1100-2650 deg/s^2 of angular
    acceleration -- 2-3 orders of magnitude past anything ever validated
    stable) that a full GPU training run was needed to notice, at the cost of
    hours of wasted compute. These checks catch the same class of mistake
    (bad geometry, or an authority regression) in milliseconds instead.
    """

    def setUp(self) -> None:
        self.source = CFG_SOURCE.read_text()
        self.layout = _extract_tuple_block(self.source, "rcs_thruster_layout")
        self.engine_dir = _extract_literal(self.source, "engine_thrust_dir_b")
        self.max_rcs_thrust_n = _extract_literal(self.source, "max_rcs_thrust_n")

    def test_eight_thrusters_with_unit_directions(self) -> None:
        self.assertEqual(len(self.layout), 8)
        for pos, direction in self.layout:
            self.assertEqual(len(pos), 3)
            self.assertAlmostEqual(_norm(direction), 1.0, places=3)

    def test_thruster_direction_is_tangential_to_its_position(self) -> None:
        # A physically-sane "tangential ring" thruster fires perpendicular to
        # its own radius vector (in the XY-plane) -- pos . dir ~= 0. A wrong
        # sign/axis mistake in the layout would show up here as a large dot
        # product instead of ~0.
        for pos, direction in self.layout:
            dot = sum(p * d for p, d in zip(pos, direction))
            self.assertAlmostEqual(dot, 0.0, places=3)

    def test_roll_torque_alternates_ccw_cw_around_the_ring(self) -> None:
        # rcs_thruster_layout's own comment (and hopper_lunar.xml's site
        # comments) claim alternating CCW/CW firing -- verify the resulting
        # torque_z actually alternates sign thruster-to-thruster, not just
        # the claim in prose.
        signs = []
        for pos, direction in self.layout:
            torque = _cross(pos, direction)
            self.assertGreater(abs(torque[2]), 1e-6, "expected a nonzero roll component")
            signs.append(torque[2] > 0)
        for i in range(len(signs)):
            self.assertNotEqual(
                signs[i], signs[(i + 1) % len(signs)], f"rcs_{i} and rcs_{(i + 1) % len(signs)} do not alternate"
            )

    def test_engine_thrust_is_pure_minus_z_so_it_yields_zero_torque(self) -> None:
        # _pre_physics_step relies on this: a force exactly along body -Z
        # passes through the body's Z-axis and contributes zero torque, so
        # the main engine's torque is hardcoded to zero rather than computed
        # (see "torque_w = torch.zeros_like(force_w)" in lunar_lander_env.py).
        # If this ever stops being exactly (0, 0, -1), that shortcut becomes
        # physically wrong.
        self.assertEqual(tuple(self.engine_dir), (0.0, 0.0, -1.0))

    def test_rcs_authority_is_not_absurdly_oversized(self) -> None:
        # Regression guard for the actual bug found on 2026-09-22: at the
        # original max_rcs_thrust_n=1.2 N, a single thruster produced
        # ~19.2 rad/s^2 (~1100 deg/s^2) of angular acceleration -- two to
        # three orders of magnitude past anything ever validated stable
        # under this project's TVC history (where even a ~1.5x floor
        # increase collapsed success_rate). Ixx=Iyy=0.025 kg*m^2 is the
        # hopper_lunar.xml diaginertia value.
        Ixx = 0.025
        max_single_torque = max(
            max(abs(t) for t in _cross(pos, tuple(d * self.max_rcs_thrust_n for d in direction))[:2])
            for pos, direction in self.layout
        )
        max_single_angaccel_deg_s2 = math.degrees(max_single_torque / Ixx)
        self.assertLess(
            max_single_angaccel_deg_s2,
            200.0,
            f"single-thruster angular accel {max_single_angaccel_deg_s2:.1f} deg/s^2 is implausibly large "
            "for an untrained policy's exploration noise -- see this test's docstring / the "
            "max_rcs_thrust_n cfg comment's derivation before raising this authority",
        )


if __name__ == "__main__":
    unittest.main()
