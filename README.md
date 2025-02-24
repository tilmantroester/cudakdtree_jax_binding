# An JAX interface to cudaKDTree
Heavily inspired by https://github.com/EiffL/JaxKDTree. 
Some improvements on `cudaKDTree`-side:
- Support for periodic boundary conditions, both in stack-based and stack-free algorithms

New wrapper:
- Tracks ordering of points (`cudaKDTree` shuffles the input points, even in read-only jax arrays!)
- Exposes options and algorithms (eg the faster `cct` traversal algorithm) in `cudaKDTree`
- Uses new JAX FFI.

## Installation
Requirements: currently requires jax main until 0.5.1 is out.

`pip install .`

## Debug build
`pip install . --config-settings=cmake.build-type=Debug -v`

## TODO
- cudaKDTree: the stack-free implementation already tracks the node bounding boxes. This should allow early exit of the traversal, similar to the `cct` algorithm.
- Check if pre-sorting the points speed things up