"""
Comprehensive tests for starting_conformations module
"""

import time
import numpy as np
import pytest
from polychrom import starting_conformations


class TestGrowCubic:
    """Tests for grow_cubic function"""

    def test_grow_cubic_correct_length(self):
        """Test that grow_cubic produces conformations of the correct length"""
        # Test different sizes
        test_cases = [
            (100, 20, "standard"),
            (200, 30, "extended"),
            (101, 25, "linear"),  # Odd number for linear
            (500, 40, "standard"),
        ]

        for N, boxSize, method in test_cases:
            polymer = starting_conformations.grow_cubic(N, boxSize, method)
            assert len(polymer) == N, f"Expected length {N}, got {len(polymer)} for {method}"
            assert polymer.shape == (N, 3), f"Expected shape ({N}, 3), got {polymer.shape}"

    def test_grow_cubic_stays_in_box(self):
        """Test that grow_cubic doesn't overshoot the box boundaries"""
        N = 100
        boxSize = 20

        for method in ["standard", "extended", "linear"]:
            polymer = starting_conformations.grow_cubic(N, boxSize, method)

            # Check that all coordinates are within bounds
            # Linear can stick out by 1
            if method == "linear":
                assert np.all(polymer >= -1), f"Polymer goes below -1 for {method}"
                assert np.all(polymer <= boxSize), f"Polymer exceeds boxSize for {method}"
            else:
                assert np.all(polymer >= 0), f"Polymer goes below 0 for {method}"
                assert np.all(polymer < boxSize), f"Polymer exceeds boxSize-1 for {method}"

    def test_grow_cubic_connectivity(self):
        """Test that polymer maintains connectivity (bond length = 1)"""
        N = 100
        boxSize = 20

        for method in ["standard", "extended"]:  # Skip linear as it can have different connectivity
            polymer = starting_conformations.grow_cubic(N, boxSize, method)

            # Calculate bond vectors
            bonds = np.diff(polymer, axis=0)
            bond_lengths = np.linalg.norm(bonds, axis=1)

            # All bonds should have length 1 (on cubic lattice)
            assert np.allclose(bond_lengths, 1.0), f"Non-unit bonds found in {method}"

    def test_grow_cubic_ring_closure(self):
        """Test that rings are properly closed"""
        N = 100
        boxSize = 20

        for method in ["standard", "extended"]:
            polymer = starting_conformations.grow_cubic(N, boxSize, method)

            # Check that first and last monomers are connected
            distance = np.linalg.norm(polymer[0] - polymer[-1])
            assert distance == 1.0, f"Ring not closed properly in {method}, distance={distance}"

    def test_grow_cubic_errors(self):
        """Test error conditions"""
        # N too large for box
        with pytest.raises(ValueError, match="Steps has to be less than size"):
            starting_conformations.grow_cubic(1002, 10, "standard")

        # Odd N for rings
        with pytest.raises(ValueError, match="N has to be multiple of 2"):
            starting_conformations.grow_cubic(101, 20, "standard")

        # Invalid method
        with pytest.raises(ValueError, match="method should be"):
            starting_conformations.grow_cubic(100, 20, "invalid")

        # Polymer too short for extended method
        with pytest.raises(ValueError, match="polymer too short"):
            starting_conformations.grow_cubic(10, 20, "extended")

    def test_grow_cubic_warnings(self):
        """Test that appropriate warnings are raised"""
        with pytest.warns(UserWarning):
            # This should trigger the slow warning
            starting_conformations.grow_cubic(920, 10, "standard")

    def test_grow_cubic_no_self_intersection(self):
        """Test that the polymer doesn't self-intersect"""
        N = 100
        boxSize = 25  # Larger box to reduce chance of getting stuck

        polymer = starting_conformations.grow_cubic(N, boxSize, "standard")

        # Check for unique positions (no two monomers at same location)
        unique_positions = np.unique(polymer, axis=0)
        assert len(unique_positions) == N, f"Self-intersection detected: {N - len(unique_positions)} duplicates"


class TestCreateRandomWalk:
    """Tests for create_random_walk function"""

    def test_random_walk_length(self):
        """Test that random walk has correct length"""
        N = 100
        step_size = 1.5

        walk = starting_conformations.create_random_walk(step_size, N)
        assert len(walk) == N
        assert walk.shape == (N, 3)

    def test_random_walk_step_size(self):
        """Test that random walk has correct step size"""
        N = 100
        step_size = 2.0

        walk = starting_conformations.create_random_walk(step_size, N)

        # Calculate step sizes
        steps = np.diff(walk, axis=0)
        step_lengths = np.linalg.norm(steps, axis=1)

        assert np.allclose(step_lengths, step_size), "Step sizes don't match specified value"

    def test_random_walk_starting_point(self):
        """Test that random walk starts at origin"""
        N = 100
        step_size = 1.0

        walk = starting_conformations.create_random_walk(step_size, N)

        # First point should be at distance step_size from origin
        first_distance = np.linalg.norm(walk[0])
        assert np.isclose(first_distance, step_size), "First point not at correct distance from origin"


class TestCreateSpiral:
    """Tests for create_spiral function"""

    def test_spiral_length(self):
        """Test that spiral has correct length"""
        r1 = 10
        r2 = 13
        N = 100

        spiral = starting_conformations.create_spiral(r1, r2, N)
        assert len(spiral) == N
        assert spiral.shape == (N, 3)

    def test_spiral_connectivity(self):
        """Test that spiral maintains approximate connectivity"""
        r1 = 10
        r2 = 13
        N = 100

        spiral = starting_conformations.create_spiral(r1, r2, N)

        # Calculate bond lengths
        bonds = np.diff(spiral, axis=0)
        bond_lengths = np.linalg.norm(bonds, axis=1)

        # Bonds should be approximately 1
        assert np.all(bond_lengths < 1.5), "Some bonds too long in spiral"
        assert np.all(bond_lengths > 0.5), "Some bonds too short in spiral"


class TestConstrainedRandomWalk:
    """Tests for create_constrained_random_walk function"""

    def test_constrained_walk_length(self):
        """Test that constrained walk has correct length"""
        N = 50  # Smaller for faster test

        def always_true(p):
            return True

        walk = starting_conformations.create_constrained_random_walk(
            N, always_true, step_size=1.0
        )
        assert len(walk) == N
        assert walk.shape == (N, 3)

    def test_constrained_walk_respects_constraint(self):
        """Test that constrained walk respects the constraint function"""
        N = 50
        confinement = 10.0

        def confined(p):
            return np.linalg.norm(p) < confinement

        walk = starting_conformations.create_constrained_random_walk(
            N, confined, starting_point=(0, 0, 0)
        )

        # Check all points are within confinement
        distances = np.linalg.norm(walk, axis=1)
        assert np.all(distances < confinement), "Some points outside confinement"

    def test_constrained_walk_starting_point(self):
        """Test that constrained walk starts at specified point"""
        N = 30
        start = (5.0, 3.0, 2.0)

        def always_true(p):
            return True

        walk = starting_conformations.create_constrained_random_walk(
            N, always_true, starting_point=start
        )

        assert np.allclose(walk[0], start), "Walk doesn't start at specified point"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

    # Run benchmark
    benchmark_grow_cubic()