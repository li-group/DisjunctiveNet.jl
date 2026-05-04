using ChainRulesCore
using DiffOpt
import MathOptInterface as MOI

function ChainRulesCore.rrule(
    ::typeof(project),
    hull::ConvexHullForm,
    yhat::AbstractVector{<:Real};
    solver = nothing,
)
    result = project(hull, yhat; solver = solver)

    function pullback(dresult)
        dy = _projection_result_y_tangent(dresult, result.y)

        if result.status != MOI.OPTIMAL
            return NoTangent(), NoTangent(), zeros(Float64, length(yhat))
        end

        model = result.model
        y_var = model[:y]
        yhat_param = model[:yhat_param]

        MOI.set.(
            model,
            DiffOpt.ReverseVariablePrimal(),
            y_var,
            Float64.(dy),
        )

        DiffOpt.reverse_differentiate!(model)

        yhat_refs = [ParameterRef(yhat_param[j]) for j in eachindex(yhat)]
        raw_grad = MOI.get.(model, DiffOpt.ReverseConstraintSet(), yhat_refs)

        grad = Float64[g.value for g in raw_grad]

        return NoTangent(), NoTangent(), grad
    end

    return result, pullback
end


function ChainRulesCore.rrule(
    ::typeof(project),
    model::DisjunctiveModel,
    yhat::AbstractVector{<:Real};
    solver = nothing,
)
    hull = convex_hull_form(model)
    result = project(hull, yhat; solver = solver)

    function pullback(dresult)
        dy = _projection_result_y_tangent(dresult, result.y)

        if result.status != MOI.OPTIMAL
            return NoTangent(), NoTangent(), zeros(Float64, length(yhat))
        end

        opt_model = result.model
        y_var = opt_model[:y]
        yhat_param = opt_model[:yhat_param]

        MOI.set.(
            opt_model,
            DiffOpt.ReverseVariablePrimal(),
            y_var,
            Float64.(dy),
        )

        DiffOpt.reverse_differentiate!(opt_model)

        yhat_refs = [ParameterRef(yhat_param[j]) for j in eachindex(yhat)]
        raw_grad = MOI.get.(opt_model, DiffOpt.ReverseConstraintSet(), yhat_refs)

        grad = Float64[g.value for g in raw_grad]

        return NoTangent(), NoTangent(), grad
    end

    return result, pullback
end


function _projection_result_y_tangent(dresult, y_template)
    if dresult isa ChainRulesCore.AbstractZero
        return zeros(Float64, length(y_template))
    end

    unthunked = ChainRulesCore.unthunk(dresult)

    if unthunked isa ProjectionResult
        return Float64.(unthunked.y)
    end

    if unthunked isa NamedTuple && haskey(unthunked, :y)
        return Float64.(unthunked.y)
    end

    if unthunked isa Tuple
        return Float64.(unthunked[1])
    end

    return Float64.(unthunked)
end

function ChainRulesCore.rrule(
    layer::DisjunctiveProjectionLayer,
    yhat::AbstractVector{<:Real},
)
    y = layer(yhat)

    function pullback(dy)
        hull = convex_hull_form(layer.model)

        _, pb = ChainRulesCore.rrule(
            project,
            hull,
            yhat;
            solver = layer.config.solver,
        )

        dproject = ProjectionResult(
            Float64.(ChainRulesCore.unthunk(dy)),
            Float64[],
            MOI.OPTIMAL,
            nothing,
        )

        _, _, grad_yhat = pb(dproject)

        return NoTangent(), grad_yhat
    end

    return y, pullback
end