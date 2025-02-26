import numpy as np

import jax
import jax.ffi as ffi

from cudakdtree_jax_binding import _cudakdtree_interface

TraversalMode = _cudakdtree_interface.TraversalMode
CandidateList = _cudakdtree_interface.CandidateList

for name, target in _cudakdtree_interface.registrations().items():
    ffi.register_ffi_target(name, target, platform="CUDA")


# Compiled combinations of n_dim, k, travesal_mode, candidate_list
_supported_kdtree_configs = [
    (2, 8, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
    (2, 9, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
    (2, 16, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
    (2, 25, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
    (2, 61, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
    (2, 128, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),

    (3, 8, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
    (3, 16, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
    (3, 27, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
    (3, 33, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
    (3, 100, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
    (3, 128, TraversalMode.stack_free_bounds_tracking, CandidateList.fixed_list),
    (3, 27, TraversalMode.cct, CandidateList.fixed_list),
]

def kdtree_call(points, k: int, queries=None, box_size=None, max_radius=np.inf, 
                traversal_mode=TraversalMode.stack_free_bounds_tracking,
                candidate_list=CandidateList.fixed_list
                ):
    if box_size is None:
        box_size = ()

    if queries is None:
        queries = np.int32(0)
        n_query = points.shape[0]
    else:
        n_query = queries.shape[0]
    idx_type = np.int32
    out_type = jax.ShapeDtypeStruct((n_query, k), idx_type)

    n_dim = points.shape[1]
    if (n_dim, k, traversal_mode, candidate_list) not in _supported_kdtree_configs:
        raise ValueError(f"The combination of {n_dim=}, {k=}, {traversal_mode=}, "
                         f"{candidate_list=} is not supported. The cuda interface "
                         f"needs to be recompiled with this combination enabled.")

    return ffi.ffi_call(
        target_name="kdtree_call",
        result_shape_dtypes=out_type,
    )(
        points, queries,
        traversal_mode=np.int32(traversal_mode),
        candidate_list=np.int32(candidate_list),
        k=np.int32(k),
        max_radius=np.float32(max_radius),
        box_size=np.array(box_size, dtype=np.float32)
    )
