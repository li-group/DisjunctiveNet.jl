"""
    convex_hull_form(model::DisjunctiveModel)

Build the convex-hull scenario expansion from a user-facing model.
"""
convex_hull_form(model::DisjunctiveModel) = convex_hull_form(standard_form(model))


"""
    convex_hull_form(model::StandardDisjunctiveModel)

Build the scenario expansion needed by the convex-hull projection formulation.

Each scenario corresponds to one choice of disjunct from every disjunction.

Global constraints are stored separately because the JuMP/DiffOpt backend should
copy them into every scenario using the perspective form

    a' * y_copy[s, :] sense b * gamma[s]
"""
function convex_hull_form(model::StandardDisjunctiveModel)
    choices = _scenario_choices(model.disjunctions)

    scenarios = ConvexHullScenario[]

    for choice in choices
        local_constraints = _local_constraints_for_choice(model.disjunctions, choice)
        push!(scenarios, ConvexHullScenario(choice, local_constraints))
    end

    return ConvexHullForm(
        model.n_outputs,
        model.lb,
        model.ub,
        model.global_constraints,
        scenarios,
    )
end


"""
    num_scenarios(hull::ConvexHullForm)

Return the number of scenarios in the convex-hull expansion.
"""
num_scenarios(hull::ConvexHullForm) = length(hull.scenarios)


function _scenario_choices(disjunctions::Vector{Disjunction})
    if isempty(disjunctions)
        return [Int[]]
    end

    choices = Vector{Vector{Int}}([Int[]])

    for disjunction in disjunctions
        new_choices = Vector{Vector{Int}}()

        for existing_choice in choices
            for disjunct_index in 1:length(disjunction.disjuncts)
                push!(new_choices, vcat(existing_choice, disjunct_index))
            end
        end

        choices = new_choices
    end

    return choices
end


function _local_constraints_for_choice(
    disjunctions::Vector{Disjunction},
    choice::Vector{Int},
)
    length(choice) == length(disjunctions) ||
        throw(DimensionMismatch("Choice length must match number of disjunctions."))

    local_constraints = LinearConstraint[]

    for (j, selected_disjunct_index) in enumerate(choice)
        disjunction = disjunctions[j]

        1 <= selected_disjunct_index <= length(disjunction.disjuncts) ||
            throw(ArgumentError("Invalid disjunct choice $(selected_disjunct_index) for disjunction $(j)."))

        selected_disjunct = disjunction.disjuncts[selected_disjunct_index]

        append!(local_constraints, selected_disjunct)
    end

    return local_constraints
end