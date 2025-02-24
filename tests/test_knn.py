import pytest

import numpy as np

import jax
import jax.numpy as jnp

from cudakdtree_jax_binding.cudakdtree_interface import kdtree_call
from cudakdtree_jax_binding.cudakdtree_interface import TraversalMode
from cudakdtree_jax_binding.cudakdtree_interface import CandidateList

import scipy.spatial


def uniform_random_points(n_dim, n_point):
    key = jax.random.key(42)
    points = jax.random.uniform(key=key, shape=(n_point, n_dim))

    return points, (1.0,)*n_dim


def toy_2d_point_set():
    pos = jnp.array([
        [10, 15], [46, 63], [68, 21], [40, 33], [25, 54], [15, 43], [44, 58], [45, 40], [62, 69], [53, 67]
    ], dtype=jnp.float32)
    box_size = (70.0, 70.0)

    return pos, box_size


UNIFORM_100_2D, UNIFORM_100_2D_BOX = uniform_random_points(n_dim=2, n_point=100)
UNIFORM_1000_2D, UNIFORM_100_2D_BOX = uniform_random_points(n_dim=2, n_point=1000)

UNIFORM_100_3D, UNIFORM_100_3D_BOX = uniform_random_points(n_dim=3, n_point=100)
UNIFORM_1000_3D, UNIFORM_100_3D_BOX = uniform_random_points(n_dim=3, n_point=1000)

TOY_10_2D, TOY_10_2D_BOX = toy_2d_point_set()


def scipy_knn(points, k, box_size, max_radius):
    kdtree = scipy.spatial.KDTree(points, boxsize=box_size)
    _, scipy_idx = kdtree.query(points, k=k, distance_upper_bound=max_radius)
    return scipy_idx


@pytest.mark.parametrize(
    "points, k, max_radius, box_size, traversal_mode, candidate_list",
    [
        (TOY_10_2D, 9, np.inf, None, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
        (TOY_10_2D, 9, np.inf, TOY_10_2D_BOX, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
        (TOY_10_2D, 25, np.inf, TOY_10_2D_BOX, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
        (TOY_10_2D, 25, 30.0, TOY_10_2D_BOX, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
        (TOY_10_2D, 25, 30.0, TOY_10_2D_BOX, TraversalMode.stack_free_bounds_tracking, CandidateList.heap),

        (UNIFORM_100_2D, 9, np.inf, None, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
        (UNIFORM_100_2D, 9, np.inf, UNIFORM_100_2D_BOX, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
        (UNIFORM_100_2D, 25, np.inf, UNIFORM_100_2D_BOX, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
        (UNIFORM_100_2D, 25, np.inf, UNIFORM_100_2D_BOX, TraversalMode.stack_free_bounds_tracking, CandidateList.heap),
        (UNIFORM_100_3D, 27, np.inf, UNIFORM_100_3D_BOX, TraversalMode.stack_free_bounds_tracking, CandidateList.heap),
        (UNIFORM_100_3D, 27, np.inf, None, TraversalMode.cct, CandidateList.heap),

        (UNIFORM_1000_2D, 25, np.inf, UNIFORM_100_2D_BOX, TraversalMode.stack_free_bounds_tracking, CandidateList.heap),
        (UNIFORM_1000_3D, 27, np.inf, UNIFORM_100_3D_BOX, TraversalMode.stack_free_bounds_tracking, CandidateList.heap),
    ]
)
def test_knn(points, k, max_radius, box_size, traversal_mode, candidate_list, debug_print=False):
    cuda_idx = kdtree_call(
        points=points, k=k, queries=points,
        max_radius=max_radius,
        box_size=box_size,
        traversal_mode=traversal_mode,
        candidate_list=candidate_list
    )
    cuda_idx.block_until_ready()

    scipy_idx = scipy_knn(points, k, box_size, max_radius)

    # If n_points < k, scipy returns n_point, while cudakdtree returns -1
    scipy_idx[scipy_idx == scipy_idx.shape[0]] = -1

    if debug_print:
        [print(f"{i}: {c} - {s} {'x' if not jnp.all(c==s) else ' '}") for i, (c, s) in enumerate(zip(cuda_idx, scipy_idx))]

    assert np.all(cuda_idx == scipy_idx)


def test_jit():
    kdtree_call_jit = jax.jit(kdtree_call, static_argnames=["k", "box_size", "traversal_mode", "candidate_list"])

    cuda_idx = kdtree_call_jit(
        points=UNIFORM_100_2D, k=9, queries=UNIFORM_100_2D,
        box_size=UNIFORM_100_2D_BOX,
    )

    cuda_idx = kdtree_call_jit(
        points=UNIFORM_100_2D, k=9,
        box_size=UNIFORM_100_2D_BOX,
    )


if __name__ == "__main__":
    test_knn(UNIFORM_100_2D, 25, np.inf, UNIFORM_100_2D_BOX, TraversalMode.stack_free_bounds_tracking, CandidateList.heap, debug_print=True)