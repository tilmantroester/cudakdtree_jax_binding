import numpy as np

import jax
import jax.extend.ffi as ffi

from cudakdtree_jax_binding import _cudakdtree_interface

TravelsalMode = _cudakdtree_interface.TraversalMode

for name, target in _cudakdtree_interface.registrations().items():
    ffi.register_ffi_target(name, target)


def kdtree_call(points, k: int, queries=None, box_size=None, traversal_mode="cct"):
    if box_size is None:
        box_size = ()

    if queries is None:
        queries = np.int32(0)
        n_query = points.shape[0]
    else:
        n_query = queries.shape[0]
    idx_type = np.int32
    out_type = jax.ShapeDtypeStruct((n_query, k), idx_type)

    return ffi.ffi_call(
        target_name="kdtree_call",
        result_shape_dtypes=out_type,
    )(
        points, queries, queries, queries,
        traversal_mode=traversal_mode,
        k=np.int32(k),
        box_size=np.array(box_size, dtype=np.float32)
    )
