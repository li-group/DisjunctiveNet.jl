"""
    ProjectionConfig(; mode = :dnf_qp, gradient = :diffopt, solver = nothing)

Configuration for a disjunctive differentiable projection layer.

# Fields

- `mode`: Projection formulation. Currently intended values are `:dnf_qp` and `:milp`.
- `gradient`: Backward rule. Currently intended values are `:diffopt` and `:straight_through`.
- `solver`: Optimizer constructor, for example `HiGHS.Optimizer` or `Gurobi.Optimizer`.
- `tol`: Numerical tolerance.
- `fallback`: Behavior when projection fails. Currently intended values are `:identity` or `:error`.
"""
struct ProjectionConfig
    mode::Symbol
    gradient::Symbol
    solver::Any
    tol::Float64
    fallback::Symbol
end

function ProjectionConfig(;
    mode::Symbol = :dnf_qp,
    gradient::Symbol = :diffopt,
    solver = nothing,
    tol::Real = 1e-6,
    fallback::Symbol = :identity,
)
    config = ProjectionConfig(
        mode,
        gradient,
        solver,
        Float64(tol),
        fallback,
    )
    _validate_projection_config(config)
    return config
end


"""
    DisjunctiveModel(n_outputs)

Container for user-defined JuMP-like disjunctive constraints.

At this stage, this stores only metadata. Later it will store variables,
ordinary constraints, disjunctions, bounds, and canonicalized convex-hull data.
"""
mutable struct DisjunctiveModel
    n_outputs::Int
    metadata::Dict{Symbol, Any}

    function DisjunctiveModel(n_outputs::Integer)
        n_outputs > 0 || throw(ArgumentError("n_outputs must be positive."))
        return new(Int(n_outputs), Dict{Symbol, Any}())
    end
end


"""
    DisjunctiveProjectionLayer(model; kwargs...)

Differentiable projection layer associated with a `DisjunctiveModel`.
"""
struct DisjunctiveProjectionLayer
    model::DisjunctiveModel
    config::ProjectionConfig

    function DisjunctiveProjectionLayer(model::DisjunctiveModel, config::ProjectionConfig)
        _validate_projection_config(config)
        return new(model, config)
    end
end