# An JAX interface to cudaKDTree
Heavily inspired by https://github.com/EiffL/JaxKDTree. 
Some improvements on `cudaKDTree`-side:
- Support for periodic boundary conditions, both in stack-based and stack-free algorithms

New wrapper:
- Tracks ordering of points (`cudaKDTree` shuffles the input points, even in read-only jax arrays!)
- Exposes options and algorithms (eg the faster `cct` traversal algorithm) in `cudaKDTree`
- Uses new JAX FFI.

## Installation
Requirements: jax>=0.5.1.

`pip install .`

## Usage notes
Because `cudaKDTree` is heavily template based, only the set of combinations of spatial dimensionality, number of neighbours, traversal algorithm, and candidate list algorithm that are built at compile time are available in the wrapper.

The current set of combinations is listed [here](https://github.com/tilmantroester/cudakdtree_jax_binding/blob/main/src/cudakdtree_jax_binding/cudakdtree_interface.py#L15).


## Debug build
`pip install . --config-settings=cmake.build-type=Debug -v`

## TODO
- cudaKDTree: the stack-free implementation already tracks the node bounding boxes. This should allow early exit of the traversal, similar to the `cct` algorithm.
- Check if pre-sorting the points speed things up.
- Check where the the heap candidate list is faster than the fixed list.