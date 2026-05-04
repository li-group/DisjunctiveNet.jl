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
    LinearConstraint(a, sense, b)

A scalar affine constraint of the form

    a' * y <= b
    a' * y >= b
    a' * y == b

where `sense` is one of `<=`, `>=`, or `==`.
"""
struct LinearConstraint
    a::Vector{Float64}
    sense::Symbol
    b::Float64

    function LinearConstraint(a::AbstractVector{<:Real}, sense::Symbol, b::Real)
        sense in (:<=, :>=, :(==)) ||
            throw(ArgumentError("sense must be one of :<=, :>=, or :(==)."))
        return new(Float64.(collect(a)), sense, Float64(b))
    end
end


"""
    Disjunction(disjuncts)

A disjunction represented as a list of disjuncts.

Each disjunct is a vector of `LinearConstraint`s. For example, a two-way
disjunction has the form

    Disjunction([
        [constraint_1, constraint_2],
        [constraint_3, constraint_4],
    ])
"""
struct Disjunction
    disjuncts::Vector{Vector{LinearConstraint}}

    function Disjunction(disjuncts::Vector{Vector{LinearConstraint}})
        isempty(disjuncts) && throw(ArgumentError("A disjunction must contain at least one disjunct."))
        return new(disjuncts)
    end
end


"""
    DisjunctiveModel(n_outputs)

Container for user-defined JuMP-like disjunctive constraints.
"""
mutable struct DisjunctiveModel
    n_outputs::Int
    lb::Vector{Float64}
    ub::Vector{Float64}
    constraints::Vector{LinearConstraint}
    disjunctions::Vector{Disjunction}
    metadata::Dict{Symbol, Any}

    function DisjunctiveModel(n_outputs::Integer)
        n_outputs > 0 || throw(ArgumentError("n_outputs must be positive."))
        n = Int(n_outputs)
        return new(
            n,
            fill(-Inf, n),
            fill(Inf, n),
            LinearConstraint[],
            Disjunction[],
            Dict{Symbol, Any}(),
        )
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