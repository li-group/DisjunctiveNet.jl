using JuMP
using HiGHS
using DiffOpt
import MathOptInterface as MOI

"""
    build_projection_model(hull, yhat; solver = nothing)

Build the convex-hull projection model.

This function constructs the JuMP model but does not solve it.
"""
function build_projection_model(
    hull::ConvexHullForm,
    yhat::AbstractVector{<:Real};
    solver = nothing,
)
    length(yhat) == hull.n_outputs ||
        throw(DimensionMismatch("Expected yhat of length $(hull.n_outputs), got $(length(yhat))."))

    optimizer = solver === nothing ? HiGHS.Optimizer : solver
    model = Model(() -> DiffOpt.diff_optimizer(optimizer))
    set_silent(model)

    n = hull.n_outputs
    S = num_scenarios(hull)

    @variable(model, y[1:n])
    @variable(model, gamma[1:S] >= 0.0)
    @variable(model, y_copy[1:S, 1:n])
    @variable(model, yhat_param[1:n] in MOI.Parameter.(Float64.(yhat)))
    
    @constraint(model, sum(gamma[s] for s in 1:S) == 1.0)

    @constraint(model, [j in 1:n], y[j] == sum(y_copy[s, j] for s in 1:S))

    for s in 1:S
        for j in 1:n
            if isfinite(hull.lb[j])
                @constraint(model, y_copy[s, j] >= hull.lb[j] * gamma[s])
            end

            if isfinite(hull.ub[j])
                @constraint(model, y_copy[s, j] <= hull.ub[j] * gamma[s])
            end
        end

        for constraint in hull.global_constraints
            _add_perspective_constraint!(model, constraint, y_copy, gamma, s)
        end

        for constraint in hull.scenarios[s].local_constraints
            _add_perspective_constraint!(model, constraint, y_copy, gamma, s)
        end
    end

    @objective(model, Min, sum((y[j] - yhat_param[j])^2 for j in 1:n))

    model[:y] = y
    model[:gamma] = gamma
    model[:y_copy] = y_copy
    model[:hull] = hull
    model[:yhat_param] = yhat_param

    return model
end


function _add_perspective_constraint!(
    model::JuMP.Model,
    constraint::LinearConstraint,
    y_copy,
    gamma,
    s::Int,
)
    lhs = sum(constraint.a[j] * y_copy[s, j] for j in eachindex(constraint.a))
    rhs = constraint.b * gamma[s]

    if constraint.sense == :<=
        @constraint(model, lhs <= rhs)
    elseif constraint.sense == :>=
        @constraint(model, lhs >= rhs)
    elseif constraint.sense == :(==)
        @constraint(model, lhs == rhs)
    else
        throw(ArgumentError("Unsupported constraint sense $(constraint.sense)."))
    end

    return nothing
end


"""
    project(hull, yhat; solver = nothing)

Solve the convex-hull projection problem.
"""
function project(
    hull::ConvexHullForm,
    yhat::AbstractVector{<:Real};
    solver = nothing,
)
    model = build_projection_model(hull, yhat; solver = solver)
    optimize!(model)

    status = termination_status(model)

    if status != MOI.OPTIMAL
        return ProjectionResult(
            Float64.(collect(yhat)),
            Float64[],
            status,
            model,
        )
    end

    y = value.(model[:y])
    gamma = value.(model[:gamma])

    return ProjectionResult(
        Float64.(y),
        Float64.(gamma),
        status,
        model,
    )
end


"""
    project(model, yhat; solver = nothing)

Convenience method that accepts a user-facing `DisjunctiveModel`.
"""
function project(
    model::DisjunctiveModel,
    yhat::AbstractVector{<:Real};
    solver = nothing,
)
    hull = convex_hull_form(model)
    return project(hull, yhat; solver = solver)
end