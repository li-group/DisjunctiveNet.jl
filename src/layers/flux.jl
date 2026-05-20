using Flux

"""
    Flux.trainable(layer::DisjunctiveProjectionLayer)

Projection layers currently have no trainable parameters.

The neural network backbone is trainable; the projection layer is a
deterministic differentiable map.
"""
Flux.trainable(layer::DisjunctiveProjectionLayer) = NamedTuple()

# Register the projection layer with Flux so it behaves like a standard layer
# inside `Chain`.
Flux.@layer DisjunctiveProjectionLayer