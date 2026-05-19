function _sense_string(sense::Symbol)
    if sense == :<=
        return "<="
    elseif sense == :>=
        return ">="
    elseif sense == :(==)
        return "=="
    else
        return string(sense)
    end
end


function _linear_expr_string(a::AbstractVector)
    terms = String[]

    for (j, coeff) in enumerate(a)
        if abs(coeff) <= 1e-12
            continue
        end

        if coeff == 1
            push!(terms, "y[$j]")
        elseif coeff == -1
            push!(terms, "-y[$j]")
        else
            push!(terms, "$(coeff)*y[$j]")
        end
    end

    isempty(terms) && return "0"

    expr = join(terms, " + ")
    return replace(expr, "+ -" => "- ")
end


function _constraint_string(c::LinearConstraint)
    return "$(_linear_expr_string(c.a)) $(_sense_string(c.sense)) $(c.b)"
end


"""
    print_model(model::DisjunctiveModel; io = stdout)

Print the user-facing disjunctive model.
"""
function print_model(model::DisjunctiveModel; io::IO = stdout)
    println(io, "DisjunctiveModel")
    println(io, "  n_outputs: $(model.n_outputs)")

    println(io, "  bounds:")
    for j in 1:model.n_outputs
        println(io, "    $(model.lb[j]) <= y[$j] <= $(model.ub[j])")
    end

    println(io, "  global constraints:")
    if isempty(model.constraints)
        println(io, "    none")
    else
        for (i, c) in enumerate(model.constraints)
            println(io, "    g[$i]: $(_constraint_string(c))")
        end
    end

    println(io, "  disjunctions:")
    if isempty(model.disjunctions)
        println(io, "    none")
    else
        for (r, disj) in enumerate(model.disjunctions)
            println(io, "    disjunction[$r]:")
            for (d, disjunct) in enumerate(disj.disjuncts)
                println(io, "      disjunct[$d]:")
                for (q, c) in enumerate(disjunct)
                    println(io, "        c[$q]: $(_constraint_string(c))")
                end
            end
        end
    end

    return nothing
end


"""
    print_hull(hull::ConvexHullForm; io = stdout)

Print the DNF convex-hull scenario expansion.
"""
function print_hull(hull::ConvexHullForm; io::IO = stdout)
    println(io, "DNF ConvexHullForm")
    println(io, "  n_outputs: $(hull.n_outputs)")
    println(io, "  scenarios: $(length(hull.scenarios))")

    println(io, "  variables:")
    println(io, "    y[1:$(hull.n_outputs)]")
    println(io, "    gamma[1:$(length(hull.scenarios))]")
    println(io, "    y_copy[1:$(length(hull.scenarios)), 1:$(hull.n_outputs)]")

    println(io, "  linking constraints:")
    println(io, "    sum_s gamma[s] == 1")
    println(io, "    y[j] == sum_s y_copy[s,j] for each j")

    println(io, "  global constraints copied into every scenario:")
    if isempty(hull.global_constraints)
        println(io, "    none")
    else
        for (i, c) in enumerate(hull.global_constraints)
            println(io, "    g[$i,s]: $(_constraint_string(c)) scaled by gamma[s]")
        end
    end

    println(io, "  scenarios:")
    for (s, scenario) in enumerate(hull.scenarios)
        println(io, "    scenario[$s], choices = $(scenario.choices):")
        if isempty(scenario.local_constraints)
            println(io, "      local constraints: none")
        else
            for (q, c) in enumerate(scenario.local_constraints)
                println(io, "      c[$q,s]: $(_constraint_string(c)) scaled by gamma[$s]")
            end
        end
    end

    return nothing
end


"""
    print_hull(hull::CNFConvexHullForm; io = stdout)

Print the CNF convex-hull block representation.
"""
function print_hull(hull::CNFConvexHullForm; io::IO = stdout)
    println(io, "CNF ConvexHullForm")
    println(io, "  n_outputs: $(hull.n_outputs)")
    println(io, "  blocks: $(length(hull.blocks))")

    println(io, "  variables:")
    println(io, "    y[1:$(hull.n_outputs)]")
    println(io, "    for each block r:")
    println(io, "      gamma_r[1:num_disjuncts(r)]")
    println(io, "      y_copy_r[1:num_disjuncts(r), 1:$(hull.n_outputs)]")

    println(io, "  global constraints on y:")
    if isempty(hull.global_constraints)
        println(io, "    none")
    else
        for (i, c) in enumerate(hull.global_constraints)
            println(io, "    g[$i]: $(_constraint_string(c))")
        end
    end

    println(io, "  blocks:")
    for block in hull.blocks
        println(io, "    block[$(block.disjunction_index)]:")
        println(io, "      sum_d gamma[d] == 1")
        println(io, "      y[j] == sum_d y_copy[d,j] for each j")

        for (d, disjunct) in enumerate(block.disjuncts)
            println(io, "      disjunct[$d]:")
            println(io, "        bounds and global constraints copied with gamma[$d]")
            for (q, c) in enumerate(disjunct)
                println(io, "        c[$q]: $(_constraint_string(c)) scaled by gamma[$d]")
            end
        end
    end

    return nothing
end


"""
    print_projection_model(model::DisjunctiveModel; formulation = :dnf, io = stdout, kwargs...)

Print the lifted projection model that will be built for DNF or CNF.
"""
function print_projection_model(
    model::DisjunctiveModel;
    formulation::Symbol = :dnf,
    io::IO = stdout,
    prune_infeasible::Bool = false,
)
    print_model(model; io = io)

    println(io)
    println(io, "Lifted projection formulation:")

    if formulation == :dnf
        hull = convex_hull_form(model; prune_infeasible = prune_infeasible)
        print_hull(hull; io = io)
    elseif formulation == :cnf
        hull = cnf_hull_form(model)
        print_hull(hull; io = io)
    else
        throw(ArgumentError("Unknown formulation $(formulation). Expected :dnf or :cnf."))
    end

    return nothing
end