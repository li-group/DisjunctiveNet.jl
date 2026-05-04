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

"""
    CanonicalConstraint(A, b)

Canonical linear inequality of the form

    A * y <= b

where `A` is a matrix and `b` is a vector.
"""
struct CanonicalConstraint
    A::Matrix{Float64}
    b::Vector{Float64}

    function CanonicalConstraint(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real})
        size(A, 1) == length(b) ||
            throw(DimensionMismatch("Number of rows in A must match length of b."))
        return new(Matrix{Float64}(A), Float64.(collect(b)))
    end
end


"""
    CanonicalDisjunction(disjuncts)

Canonical disjunction represented as a vector of canonical inequality systems.

Each disjunct has the form

    A_i * y <= b_i
"""
struct CanonicalDisjunction
    disjuncts::Vector{CanonicalConstraint}

    function CanonicalDisjunction(disjuncts::Vector{CanonicalConstraint})
        isempty(disjuncts) &&
            throw(ArgumentError("A canonical disjunction must contain at least one disjunct."))
        return new(disjuncts)
    end
end


"""
    StandardDisjunctiveModel

Internal standardized representation of a `DisjunctiveModel`.

Unlike a full canonical inequality form, this preserves the original constraint
sense `:<=`, `:>=`, or `:(==)` to avoid introducing unnecessary constraints.
"""
struct StandardDisjunctiveModel
    n_outputs::Int
    lb::Vector{Float64}
    ub::Vector{Float64}
    global_constraints::Vector{LinearConstraint}
    disjunctions::Vector{Disjunction}

    function StandardDisjunctiveModel(
        n_outputs::Integer,
        lb::AbstractVector{<:Real},
        ub::AbstractVector{<:Real},
        global_constraints::Vector{LinearConstraint},
        disjunctions::Vector{Disjunction},
    )
        n = Int(n_outputs)

        length(lb) == n ||
            throw(DimensionMismatch("Lower bound length must be $(n)."))

        length(ub) == n ||
            throw(DimensionMismatch("Upper bound length must be $(n)."))

        for constraint in global_constraints
            length(constraint.a) == n ||
                throw(DimensionMismatch("Global constraint dimension must be $(n)."))
        end

        for disjunction in disjunctions
            for disjunct in disjunction.disjuncts
                for constraint in disjunct
                    length(constraint.a) == n ||
                        throw(DimensionMismatch("Disjunct constraint dimension must be $(n)."))
                end
            end
        end

        return new(
            n,
            Float64.(collect(lb)),
            Float64.(collect(ub)),
            copy(global_constraints),
            copy(disjunctions),
        )
    end
end


"""
    ConvexHullScenario

One scenario in the convex-hull expansion.

A scenario corresponds to choosing exactly one disjunct from each disjunction.
"""
struct ConvexHullScenario
    choices::Vector{Int}
    local_constraints::Vector{LinearConstraint}
end


"""
    ConvexHullForm

Convex-hull scenario expansion of a standardized disjunctive model.

Global constraints are stored separately and should be copied into every
scenario when constructing the JuMP/DiffOpt projection model.
"""
struct ConvexHullForm
    n_outputs::Int
    lb::Vector{Float64}
    ub::Vector{Float64}
    global_constraints::Vector{LinearConstraint}
    scenarios::Vector{ConvexHullScenario}
end

"""
    ProjectionResult

Result returned by the projection backend.
"""
struct ProjectionResult
    y::Vector{Float64}
    gamma::Vector{Float64}
    status::Any
    model::Any
end