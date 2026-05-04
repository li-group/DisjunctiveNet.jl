"""
    canonicalize(model::DisjunctiveModel)

Convert a user-facing `DisjunctiveModel` into a normalized internal form.

All linear constraints are converted to inequalities of the form

    A * y <= b

A `>=` constraint is multiplied by `-1`.
An equality constraint is converted into two inequalities.
"""
function canonicalize(model::DisjunctiveModel)
    global_constraints = _canonicalize_constraints(
        model.constraints,
        model.n_outputs,
    )

    canonical_disjunctions = CanonicalDisjunction[]

    for disjunction in model.disjunctions
        canonical_disjuncts = CanonicalConstraint[]

        for disjunct in disjunction.disjuncts
            push!(
                canonical_disjuncts,
                _canonicalize_constraints(disjunct, model.n_outputs),
            )
        end

        push!(canonical_disjunctions, CanonicalDisjunction(canonical_disjuncts))
    end

    return CanonicalDisjunctiveModel(
        model.n_outputs,
        model.lb,
        model.ub,
        global_constraints,
        canonical_disjunctions,
    )
end


function _canonicalize_constraints(
    constraints::Vector{LinearConstraint},
    n_outputs::Integer,
)
    rows = Vector{Vector{Float64}}()
    rhs = Float64[]

    for constraint in constraints
        _append_canonical_constraint!(rows, rhs, constraint, Int(n_outputs))
    end

    if isempty(rows)
        A = zeros(Float64, 0, Int(n_outputs))
    else
        A = reduce(vcat, reshape.(rows, 1, :))
    end

    return CanonicalConstraint(A, rhs)
end


function _append_canonical_constraint!(
    rows::Vector{Vector{Float64}},
    rhs::Vector{Float64},
    constraint::LinearConstraint,
    n_outputs::Int,
)
    length(constraint.a) == n_outputs ||
        throw(DimensionMismatch("Expected constraint dimension $(n_outputs), got $(length(constraint.a))."))

    if constraint.sense == :<=
        push!(rows, copy(constraint.a))
        push!(rhs, constraint.b)
    elseif constraint.sense == :>=
        push!(rows, -copy(constraint.a))
        push!(rhs, -constraint.b)
    elseif constraint.sense == :(==)
        push!(rows, copy(constraint.a))
        push!(rhs, constraint.b)

        push!(rows, -copy(constraint.a))
        push!(rhs, -constraint.b)
    else
        throw(ArgumentError("Unsupported constraint sense $(constraint.sense)."))
    end

    return nothing
end