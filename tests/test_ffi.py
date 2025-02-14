import jax.numpy as jnp
import numpy as np

from cudakdtree_jax_binding.ffi import test_call
from cudakdtree_jax_binding.ffi import TravelsalMode

result = test_call(
    jnp.arange(16, dtype=jnp.float32).reshape(-1, 2), 
    jnp.arange(8, dtype=jnp.float32).reshape(-1, 2),
    jnp.arange(4, dtype=jnp.float32),
    optional_input=jnp.arange(8, dtype=jnp.float32).reshape(-1, 2),
    # optional_input=None,
    array_attr=np.array((1.0, 2.0), dtype=np.float32)
    
)

# result = kdtree_call(
#     points=jnp.arange(16, dtype=jnp.float32).reshape(-1, 2),
#     # queries=np.int32(0),
#     queries=jnp.arange(8, dtype=jnp.float32).reshape(-1, 2),
#     k=np.int32(4),
#     box_size=(1, 2, 3),
#     traversal_mode=np.int32(TravelsalMode.stack_free)
# )
# print(f"{result=}")

# print(TravelsalMode, TravelsalMode.cct)
# print(int(TravelsalMode.cct))