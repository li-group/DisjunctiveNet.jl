"""
    DisjunctiveProjectionLayer(model; mode = :dnf_qp, gradient = :diffopt, solver = nothing, kwargs...)

Create a differentiable projection layer from a `DisjunctiveModel`.
"""
function DisjunctiveProjectionLayer(
    model::DisjunctiveModel;
    mode::Symbol = :dnf_qp,
    gradient::Symbol = mode == :milp ? :straight_through : :diffopt,
    solver = nothing,
    tol::Real = 1e-6,
    fallback::Symbol = :identity,
)
    config = ProjectionConfig(
        mode = mode,
        gradient = gradient,
        solver = solver,
        tol = Float64(tol),
        fallback = fallback,
    )
    return DisjunctiveProjectionLayer(model, config)
end


"""
    projection_mode(layer::DisjunctiveProjectionLayer)

Return the projection mode used by the layer.
"""
projection_mode(layer::DisjunctiveProjectionLayer) = layer.config.mode


"""
    (layer::DisjunctiveProjectionLayer)(yhat)

Placeholder call overload.

The actual projection implementation will be added later.
"""
function (layer::DisjunctiveProjectionLayer)(yhat)
    if layer.config.fallback == :identity
        return yhat
    else
        error("Projection call is not implemented yet.")
    end
end