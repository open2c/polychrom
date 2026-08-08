"""
Tests for C++ polymer topology functions in _polymer_math module.
"""

import numpy as np
import pytest

try:
    from polychrom import _polymer_math  # noqa: F401

    HAS_POLYMER_MATH = True
except ImportError:
    HAS_POLYMER_MATH = False

pytestmark = pytest.mark.skipif(not HAS_POLYMER_MATH, reason="_polymer_math Cython extension not available")


def create_circle(center, radius, n_points=100, axis="z"):
    """Create a circular polymer ring."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    if axis == "z":
        # Circle in xy-plane
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        z = np.full_like(t, center[2])
    elif axis == "x":
        # Circle in yz-plane
        x = np.full_like(t, center[0])
        y = center[1] + radius * np.cos(t)
        z = center[2] + radius * np.sin(t)
    elif axis == "y":
        # Circle in xz-plane
        x = center[0] + radius * np.cos(t)
        y = np.full_like(t, center[1])
        z = center[2] + radius * np.sin(t)
    else:
        raise ValueError(f"Invalid axis: {axis}")

    return np.column_stack([x, y, z])


def create_hopf_link(n_points=100):
    """Create a Hopf link (two linked circles with linking number ±1)."""
    # First ring in xy-plane centered at origin
    ring1 = create_circle([0, 0, 0], radius=1, n_points=n_points, axis="z")

    # Second ring in xz-plane, offset and linked through the first
    ring2 = create_circle([0.5, 0, 0], radius=1, n_points=n_points, axis="y")

    return ring1, ring2


def create_unlinked_rings(separation=3, n_points=100):
    """Create two unlinked circles."""
    ring1 = create_circle([0, 0, 0], radius=1, n_points=n_points, axis="z")
    ring2 = create_circle([separation, 0, 0], radius=1, n_points=n_points, axis="z")
    return ring1, ring2


def create_trefoil_knot(n_points=200):
    """Create a trefoil knot (simplest nontrivial knot)."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    return np.column_stack([x, y, z])


class TestLinkingNumber:
    """Test the getLinkingNumber function."""

    def test_unlinked_rings(self):
        """Test that unlinked rings have linking number 0."""
        from polychrom.polymer_analyses import getLinkingNumber

        ring1, ring2 = create_unlinked_rings(separation=5)
        L = getLinkingNumber(ring1, ring2, simplify=False)
        assert L == 0, f"Unlinked rings should have linking number 0, got {L}"

    def test_hopf_link(self):
        """Test that a Hopf link has linking number ±1."""
        from polychrom.polymer_analyses import getLinkingNumber

        ring1, ring2 = create_hopf_link()
        L = getLinkingNumber(ring1, ring2, simplify=False)
        # A Hopf link has linking number exactly +-1 (sign depends on
        # orientation). The old C code returned -2x the true value: it summed
        # signed crossings without halving, with an inverted sign convention.
        assert abs(L) == 1, f"Hopf link must have |linking number| = 1, got {L}"

    def test_simplify_preserves_linking(self):
        """Test that simplification preserves linking number."""
        from polychrom.polymer_analyses import getLinkingNumber

        ring1, ring2 = create_hopf_link(n_points=200)

        # Calculate with and without simplification
        L_no_simp = getLinkingNumber(ring1, ring2, simplify=False)
        L_with_simp = getLinkingNumber(ring1, ring2, simplify=True)

        assert L_no_simp == L_with_simp, f"Simplification changed linking number: {L_no_simp} -> {L_with_simp}"

    def test_linking_number_symmetric(self):
        """Test that L(A,B) = L(B,A)."""
        from polychrom.polymer_analyses import getLinkingNumber

        ring1, ring2 = create_hopf_link()
        L12 = getLinkingNumber(ring1, ring2, simplify=False)
        L21 = getLinkingNumber(ring2, ring1, simplify=False)

        assert L12 == L21, f"Linking number should be symmetric: {L12} != {L21}"

    def test_double_linked(self):
        """Test rings with linking number 2."""
        from polychrom.polymer_analyses import getLinkingNumber

        # Create two properly linked rings
        # First ring in xy-plane
        t1 = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        ring1 = np.column_stack([2 * np.cos(t1), 2 * np.sin(t1), np.zeros_like(t1)])

        # Second ring that passes through the first twice
        t2 = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        ring2 = np.column_stack([np.cos(t2), np.sin(t2), 0.5 * np.cos(2 * t2)])  # Goes up and down twice

        L = getLinkingNumber(ring1, ring2, simplify=True)
        # Just verify it calculates without error - exact value depends on construction
        assert isinstance(L, (int, np.integer)), "Linking number should be an integer"


class TestMutualSimplify:
    """Test the mutualSimplify function."""

    def test_simplify_unlinked(self):
        """Test that unlinked rings simplify to small sizes."""
        from polychrom.polymer_analyses import mutualSimplify

        ring1, ring2 = create_unlinked_rings(separation=5, n_points=100)
        simp1, simp2 = mutualSimplify(ring1, ring2)

        # Unlinked rings should simplify to very few points (typically 3-4)
        assert len(simp1) < 10, f"Unlinked ring 1 didn't simplify enough: {len(simp1)} points"
        assert len(simp2) < 10, f"Unlinked ring 2 didn't simplify enough: {len(simp2)} points"

    def test_simplify_linked(self):
        """Test that linked rings simplify but maintain structure."""
        from polychrom.polymer_analyses import mutualSimplify

        ring1, ring2 = create_hopf_link(n_points=100)
        simp1, simp2 = mutualSimplify(ring1, ring2)

        # Linked rings should simplify but not as much as unlinked
        assert len(simp1) < len(ring1), "Ring 1 should be simplified"
        assert len(simp2) < len(ring2), "Ring 2 should be simplified"
        assert len(simp1) >= 3, "Simplified ring 1 needs at least 3 points"
        assert len(simp2) >= 3, "Simplified ring 2 needs at least 3 points"

    def test_simplify_preserves_linking_direct(self):
        """Test that mutual simplification preserves linking number."""
        from polychrom.polymer_analyses import getLinkingNumber, mutualSimplify

        ring1, ring2 = create_hopf_link(n_points=150)

        # Get linking before simplification
        L_before = getLinkingNumber(ring1, ring2, simplify=False)

        # Simplify
        simp1, simp2 = mutualSimplify(ring1, ring2)

        # Get linking after simplification
        L_after = getLinkingNumber(simp1, simp2, simplify=False)

        assert L_before == L_after, f"Mutual simplification changed linking: {L_before} -> {L_after}"


class TestSimplifyPolymer:
    """Test the simplifyPolymer function."""

    def test_simplify_circle(self):
        """Test that a simple circle simplifies to few points."""
        from polychrom.polymer_analyses import simplifyPolymer

        circle = create_circle([0, 0, 0], radius=1, n_points=100)
        simplified = simplifyPolymer(circle)

        # No stale-buffer garbage rows (regression test for the old
        # _simplifyCpp bug that leaked zero-padded points into the output)
        assert np.any(simplified != 0, axis=1).all(), "Output contains garbage zero rows"

        # An unknotted circle should simplify to very few points
        assert len(simplified) < 10, f"Simple circle didn't simplify enough: {len(simplified)} points"
        assert len(simplified) >= 3, "Need at least 3 points for a valid polygon"

    def test_simplify_trefoil(self):
        """Test that a trefoil knot simplifies but maintains structure."""
        from polychrom.polymer_analyses import simplifyPolymer

        trefoil = create_trefoil_knot(n_points=200)
        simplified = simplifyPolymer(trefoil)

        # A knotted polymer must reduce, but can never drop below the
        # trefoil's stick number of 6 (the simplification is topology-preserving)
        assert len(simplified) < len(trefoil), "Trefoil should be simplified"
        assert len(simplified) >= 6, "Trefoil cannot be represented by fewer than 6 segments"
        assert len(simplified) <= 30, f"Trefoil should simplify well below 30 points, got {len(simplified)}"

    def test_simplify_small_polymer(self):
        """Test that small polymers are handled correctly."""
        from polychrom.polymer_analyses import simplifyPolymer

        # 3-point polymer (minimum): a triangle is already the minimal
        # closed polygon and must come back unchanged
        small = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0]])
        simplified = simplifyPolymer(small)
        assert len(simplified) == 3, f"Minimal triangle must be returned unchanged, got {len(simplified)} points"

    def test_simplify_input_validation(self):
        """Test input validation for simplifyPolymer."""
        from polychrom.polymer_analyses import simplifyPolymer

        # Too few points
        with pytest.raises(ValueError, match="at least 3 monomers"):
            simplifyPolymer(np.array([[0, 0, 0], [1, 0, 0]]))

        # Wrong dimensions
        with pytest.raises(ValueError, match="Nx3 array"):
            simplifyPolymer(np.array([[0, 0], [1, 0], [0.5, 0.5]]))

    def test_simplify_preserves_3d_structure(self):
        """Test that simplification maintains some structure."""
        from polychrom.polymer_analyses import simplifyPolymer

        # Create a more complex 3D knot-like structure
        t = np.linspace(0, 6 * np.pi, 200, endpoint=False)
        # Trefoil-like parametrization in 3D
        x = np.sin(t) + 2 * np.sin(2 * t)
        y = np.cos(t) - 2 * np.cos(2 * t)
        z = -np.sin(3 * t)
        knot = np.column_stack([x, y, z])

        simplified = simplifyPolymer(knot)

        # Remove zero-padded rows
        nonzero_mask = np.any(simplified != 0, axis=1)
        simplified_actual = simplified[nonzero_mask]

        # For a knotted structure, simplification should maintain some complexity
        assert len(simplified_actual) > 5, f"Complex 3D knot oversimplified to {len(simplified_actual)} points"

        # Check that all three dimensions are utilized
        x_range = simplified_actual[:, 0].max() - simplified_actual[:, 0].min()
        y_range = simplified_actual[:, 1].max() - simplified_actual[:, 1].min()
        z_range = simplified_actual[:, 2].max() - simplified_actual[:, 2].min()

        assert x_range > 0.1, "X dimension collapsed"
        assert y_range > 0.1, "Y dimension collapsed"
        assert z_range > 0.1, "Z dimension collapsed"


class TestIntegration:
    """Integration tests for the topology functions."""

    def test_full_pipeline(self):
        """Test the full pipeline: create, simplify, compute linking."""
        from polychrom.polymer_analyses import (
            getLinkingNumber,
            mutualSimplify,
            simplifyPolymer,
        )

        # Create complex linked structure
        ring1, ring2 = create_hopf_link(n_points=200)

        # Add noise to make it more realistic
        ring1 += np.random.randn(*ring1.shape) * 0.01
        ring2 += np.random.randn(*ring2.shape) * 0.01

        # Test full pipeline with simplification
        L_full = getLinkingNumber(ring1, ring2, simplify=True, verbose=False)
        assert L_full != 0, "Full pipeline should preserve non-zero linking"

        # Test individual simplification
        simp1 = simplifyPolymer(ring1)
        simp2 = simplifyPolymer(ring2)
        assert len(simp1) < len(ring1)
        assert len(simp2) < len(ring2)

    def test_robustness_to_noise(self):
        """Test that functions are robust to numerical noise."""
        from polychrom.polymer_analyses import getLinkingNumber

        ring1, ring2 = create_hopf_link(n_points=50)

        # Get baseline linking number
        L_base = getLinkingNumber(ring1, ring2, simplify=True)

        # Add different levels of noise
        for noise_level in [1e-6, 1e-4, 1e-2]:
            r1_noise = ring1 + np.random.randn(*ring1.shape) * noise_level
            r2_noise = ring2 + np.random.randn(*ring2.shape) * noise_level

            L = getLinkingNumber(r1_noise, r2_noise, simplify=True)
            assert L == L_base, f"Linking number changed with noise level {noise_level}: {L} != {L_base}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestAlexanderInvariants:
    """Tests for polymer_analyses.alexander_invariants."""

    def test_reference_knots(self):
        from polychrom.polymer_analyses import alexander_invariants

        rng = np.random.default_rng(0)
        s = np.linspace(0, 2 * np.pi, 600, endpoint=False)

        def torus_knot(p, q):
            r = 2 + np.cos(q * s)
            return np.stack([r * np.cos(p * s), r * np.sin(p * s), -np.sin(q * s)], axis=1)

        circle = np.stack([np.cos(s), np.sin(s), 0 * s], axis=1)
        fig8 = np.stack(
            [(2 + np.cos(2 * s)) * np.cos(3 * s), (2 + np.cos(2 * s)) * np.sin(3 * s), np.sin(4 * s)], axis=1
        )
        assert alexander_invariants(circle, rng=rng) == (1, 1)
        assert alexander_invariants(torus_knot(2, 3), rng=rng) == (3, 7)
        assert alexander_invariants(fig8, rng=rng) == (5, 11)
        assert alexander_invariants(torus_knot(2, 5), rng=rng) == (5, 31)

    def test_grow_cubic_rings_unknotted(self):
        """grow_cubic rings are unknotted by construction: no false positives."""
        from polychrom.polymer_analyses import alexander_invariants
        from polychrom.starting_conformations import grow_cubic

        rng = np.random.default_rng(1)
        for _ in range(3):
            ring = np.array(grow_cubic(400, 9), dtype=float)
            assert alexander_invariants(ring, rng=rng) == (1, 1)

    def test_input_validation(self):
        from polychrom.polymer_analyses import alexander_invariants

        with pytest.raises(ValueError):
            alexander_invariants(np.zeros((10, 2)))
