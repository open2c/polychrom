import pytest
import numpy as np
from polychrom import starting_conformations


def Jun_30_2022_create_constrained_random_walk(N, constraint_f, starting_point=(0, 0, 0), step_size=1.0):
    """
    Creates a constrained freely joined chain of length N with step step_size.
    Each step of a random walk is tested with the constraint function and is
    rejected if the tried step lies outside of the constraint.
    This function is much less efficient than create_random_walk().
    Parameters
    ----------
    N : int
        The number of steps
    constraint_f : function((float, float, float))
        The constraint function.
        Must accept a tuple of 3 floats with the tentative position of a particle
        and return True if the new position is accepted and False is it is forbidden.
    starting_point : a tuple of (float, float, float)
        The starting point of a random walk.
    step_size: float
        The size of a step of the random walk.
    """

    i = 1
    j = N
    out = np.full((N, 3), np.nan)
    out[0] = starting_point

    while i < N:
        if j == N:
            theta, u = starting_conformations._random_points_sphere(N).T
            dx = step_size * np.sqrt(1.0 - u * u) * np.cos(theta)
            dy = step_size * np.sqrt(1.0 - u * u) * np.sin(theta)
            dz = step_size * u
            d = np.vstack([dx, dy, dz]).T
            j = 0

        new_p = out[i - 1] + d[j]

        if constraint_f(new_p):
            out[i] = new_p
            i += 1

        j += 1

    return out


N, step, confinement, polar_fixed = 1000, 2.8, 10, np.pi / 2


class Correct_outputtest:
    def test_for_correct_run_all_allowed(self):
        def alwaystrue(new_p):
            return True

        polymer = starting_conformations.create_constrained_random_walk(N, alwaystrue, step_size=step)
        increments = [polymer[i + 1] - polymer[i] for i in range(len(polymer) - 1)]

        self.assertTrue(
            np.all(np.isclose(np.linalg.norm(increments, axis=1), step)),
            "Steps with default input are not as specified",
        )

    def test_for_correct_run_confined(self):
        def confined(new_p):
            return np.linalg.norm(new_p) < confinement

        polymer = starting_conformations.create_constrained_random_walk(N, confined)
        increments = [polymer[i + 1] - polymer[i] for i in range(len(polymer) - 1)]

        # Check that the conformation does not go outside the allowed region
        assert np.all(np.sqrt(np.sum(polymer**2, 1)) < confinement)


class Same_output_as_old_codetest:
    def test_for_same_output_with_allTrueconstraint_as_Jun_30_2022(self):
        def alwaystrue(new_p):
            return True

        np.random.seed(42)
        p_old = Jun_30_2022_create_constrained_random_walk(N, alwaystrue)
        np.random.seed(42)
        p_new = starting_conformations.create_constrained_random_walk(N, alwaystrue)

        # Check that the output is not different compared to old random walk starting conf for all true constraint function
        assert np.all(p_old == p_new)

    def test_for_same_output_with_sphericalconstraint_as_Jun_30_2022(self):
        def confined(new_p):
            return np.linalg.norm(new_p) < confinement

        np.random.seed(42)
        p_old = Jun_30_2022_create_constrained_random_walk(N, confined)
        np.random.seed(42)
        p_new = starting_conformations.create_constrained_random_walk(N, confined)

        # Check that output is not different for all spherical constraint function
        assert np.all(p_old == p_new)


class New_addition_test:
    def test_for_correct_angle_fixing_always_true(self):
        def alwaystrue(new_p):
            return True

        polymer = starting_conformations.create_constrained_random_walk(N, alwaystrue, polar_fixed=polar_fixed)
        angles = np.arccos(
            [np.dot(polymer[i + 2] - polymer[i + 1], polymer[i + 1] - polymer[i]) for i in range(len(polymer) - 2)]
        )

        # Check that the angles are correct
        assert np.all(np.isclose(angles, polar_fixed))

    def test_for_correct_angle_fixing_with_confinement(self):
        def confined(new_p):
            return np.linalg.norm(new_p) < confinement

        polymer = starting_conformations.create_constrained_random_walk(N, confined, polar_fixed=polar_fixed)
        angles = np.arccos(
            [np.dot(polymer[i + 2] - polymer[i + 1], polymer[i + 1] - polymer[i]) for i in range(len(polymer) - 2)]
        )

        # Check that the angles are correct
        assert np.all(np.isclose(angles, polar_fixed))

    def test_for_right_length_noconstraint(self):
        def alwaystrue(new_p):
            return True

        polymer = starting_conformations.create_constrained_random_walk(
            N, alwaystrue, step_size=step, polar_fixed=polar_fixed
        )
        increments = [polymer[i + 1] - polymer[i] for i in range(len(polymer) - 1)]

        # Check that steps with default input are as specified
        assert np.all(np.isclose(np.linalg.norm(increments, axis=1), step))

    def test_for_right_length_constraint(self):
        def confined(new_p):
            return np.linalg.norm(new_p) < confinement

        polymer = starting_conformations.create_constrained_random_walk(
            N, confined, step_size=step, polar_fixed=polar_fixed
        )
        increments = [polymer[i + 1] - polymer[i] for i in range(len(polymer) - 1)]

        # Check that steps with default input are as specified
        assert np.all(np.isclose(np.linalg.norm(increments, axis=1), step))
