using Test
using Zygote
using Flux
using DisjunctiveDifferentiableLayers
import MathOptInterface as MOI

@testset "DisjunctiveDifferentiableLayers.jl" begin
    @testset "DisjunctiveModel" begin
        dm = DisjunctiveModel(3)

        @test output_dimension(dm) == 3
        @test dm.metadata isa Dict{Symbol, Any}

        @test_throws ArgumentError DisjunctiveModel(0)
        @test_throws ArgumentError DisjunctiveModel(-1)
    end

    @testset "ProjectionConfig" begin
        cfg = ProjectionConfig()
        @test cfg.mode == :dnf_qp
        @test cfg.gradient == :diffopt
        @test cfg.fallback == :identity

        @test ProjectionConfig().formulation == :dnf
        @test ProjectionConfig(formulation = :cnf).formulation == :cnf
        @test_throws ArgumentError ProjectionConfig(formulation = :bad_formulation)

        @test_throws ArgumentError ProjectionConfig(mode = :bad_mode)
        @test_throws ArgumentError ProjectionConfig(gradient = :bad_gradient)
        @test_throws ArgumentError ProjectionConfig(fallback = :bad_fallback)
        @test_throws ArgumentError ProjectionConfig(mode = :milp, gradient = :diffopt)
    end

    @testset "Modeling interface" begin
        dm = DisjunctiveModel(3)

        set_bounds!(dm, lower = [0.0, 0.0, 0.0], upper = [1.0, 2.0, 3.0])
        @test lower_bounds(dm) == [0.0, 0.0, 0.0]
        @test upper_bounds(dm) == [1.0, 2.0, 3.0]

        c1 = add_linear_constraint!(dm, [0.0, 0.0, 1.0], :>=, 0.2)
        @test c1 isa LinearConstraint
        @test length(linear_constraints(dm)) == 1

        d1 = [
            LinearConstraint([1.0, 0.0, 0.0], :<=, 0.4),
            LinearConstraint([0.0, 1.0, 0.0], :<=, 0.7),
        ]

        d2 = [
            LinearConstraint([1.0, 0.0, 0.0], :>=, 0.6),
            LinearConstraint([0.0, 1.0, 0.0], :>=, 0.3),
        ]

        disj = add_disjunction!(dm, d1, d2)
        @test disj isa Disjunction
        @test length(disjunctions(dm)) == 1
        @test length(disjunctions(dm)[1].disjuncts) == 2

        @test_throws DimensionMismatch add_linear_constraint!(dm, [1.0, 2.0], :<=, 1.0)
        @test_throws DimensionMismatch set_bounds!(dm, lower = [0.0], upper = [1.0])
        @test_throws ArgumentError set_bounds!(dm, lower = [1.0, 0.0, 0.0], upper = [0.0, 1.0, 1.0])
        @test_throws ArgumentError LinearConstraint([1.0, 0.0, 0.0], :bad_sense, 1.0)
    end

    @testset "Standard form and convex hull" begin
        dm = DisjunctiveModel(3)

        set_bounds!(
            dm,
            lower = [0.0, 0.0, 0.0],
            upper = [1.0, 1.0, 1.0],
        )

        add_linear_constraint!(dm, [0.0, 0.0, 1.0], :>=, 0.2)
        add_linear_constraint!(dm, [1.0, 1.0, 0.0], :<=, 1.2)
        add_linear_constraint!(dm, [1.0, -1.0, 0.0], :(==), 0.0)

        d1 = [
            LinearConstraint([1.0, 0.0, 0.0], :<=, 0.4),
        ]

        d2 = [
            LinearConstraint([1.0, 0.0, 0.0], :>=, 0.6),
        ]

        add_disjunction!(dm, d1, d2)

        sm = standard_form(dm)

        @test sm.n_outputs == 3
        @test sm.lb == [0.0, 0.0, 0.0]
        @test sm.ub == [1.0, 1.0, 1.0]

        @test length(sm.global_constraints) == 3
        @test sm.global_constraints[1].sense == :>=
        @test sm.global_constraints[2].sense == :<=
        @test sm.global_constraints[3].sense == :(==)

        @test length(sm.disjunctions) == 1
        @test length(sm.disjunctions[1].disjuncts) == 2

        hull = convex_hull_form(sm; prune_infeasible=false)

        @test hull.n_outputs == 3
        @test hull.global_constraints == sm.global_constraints
        @test num_scenarios(hull) == 2

        @test hull.scenarios[1].choices == [1]
        @test length(hull.scenarios[1].local_constraints) == 1
        @test hull.scenarios[1].local_constraints[1].sense == :<=
        @test hull.scenarios[1].local_constraints[1].b == 0.4

        @test hull.scenarios[2].choices == [2]
        @test length(hull.scenarios[2].local_constraints) == 1
        @test hull.scenarios[2].local_constraints[1].sense == :>=
        @test hull.scenarios[2].local_constraints[1].b == 0.6
    end

    @testset "Scenario pruning" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        # Global equality.
        add_linear_constraint!(dm, [1.0, 1.0], :(==), 1.0)

        # Disjunction 1:
        # y1 <= 0.25 OR y1 >= 0.75
        add_disjunction!(
            dm,
            [LinearConstraint([1.0, 0.0], :<=, 0.25)],
            [LinearConstraint([1.0, 0.0], :>=, 0.75)],
        )

        # Disjunction 2:
        # y2 <= 0.25 OR y2 >= 0.75
        add_disjunction!(
            dm,
            [LinearConstraint([0.0, 1.0], :<=, 0.25)],
            [LinearConstraint([0.0, 1.0], :>=, 0.75)],
        )

        sm = standard_form(dm)

        hull_unpruned = convex_hull_form(
            sm;
            prune_infeasible = false,
        )

        @test num_scenarios(hull_unpruned) == 4

        hull_pruned = convex_hull_form(
            sm;
            prune_infeasible = true,
            interior_tol = 1e-7,
        )

        # Two scenarios are infeasible:
        # y1 <= 0.25, y2 <= 0.25 conflicts with y1 + y2 == 1.
        # y1 >= 0.75, y2 >= 0.75 conflicts with y1 + y2 == 1.
        @test num_scenarios(hull_pruned) == 2

        choices = sort(hull_pruned.scenarios .|> s -> s.choices)

        @test choices == [[1, 2], [2, 1]]
    end

    @testset "Projection backend" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        # Global constraint: y1 + y2 >= 0.8
        add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

        # Disjunction:
        # either y1 <= 0.25
        # or     y1 >= 0.75
        d1 = [
            LinearConstraint([1.0, 0.0], :<=, 0.25),
        ]

        d2 = [
            LinearConstraint([1.0, 0.0], :>=, 0.75),
        ]

        add_disjunction!(dm, d1, d2)

        yhat = [0.5, 0.1]

        result = project(dm, yhat)

        @test result.status == MOI.OPTIMAL
        @test length(result.y) == 2
        @test length(result.gamma) == 2

        # Bounds
        @test result.y[1] >= -1e-6
        @test result.y[1] <= 1.0 + 1e-6
        @test result.y[2] >= -1e-6
        @test result.y[2] <= 1.0 + 1e-6

        # Global constraint
        @test result.y[1] + result.y[2] >= 0.8 - 1e-6

        # Convex hull of y1 <= 0.25 OR y1 >= 0.75 over [0,1] is actually [0,1],
        # so y1 may remain near 0.5. This is expected for convex-hull relaxation.
        @test isapprox(sum(result.gamma), 1.0; atol = 1e-6)

        layer = DisjunctiveProjectionLayer(dm)
        yproj = layer(yhat)

        @test length(yproj) == 2
        @test yproj[1] + yproj[2] >= 0.8 - 1e-6
    end

    @testset "Differentiation" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        # Simple global constraint: y1 + y2 >= 0.8
        add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

        d1 = [
            LinearConstraint([1.0, 0.0], :<=, 0.25),
        ]

        d2 = [
            LinearConstraint([1.0, 0.0], :>=, 0.75),
        ]

        add_disjunction!(dm, d1, d2)

        layer = DisjunctiveProjectionLayer(dm)

        yhat = [0.5, 0.1]

        grad = Zygote.gradient(y -> sum(layer(y)), yhat)[1]

        @test grad !== nothing
        @test length(grad) == 2
        @test all(isfinite, grad)
    end

    @testset "Flux integration" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        # Global constraint: y1 + y2 >= 0.8
        add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

        d1 = [
            LinearConstraint([1.0, 0.0], :<=, 0.25),
        ]

        d2 = [
            LinearConstraint([1.0, 0.0], :>=, 0.75),
        ]

        add_disjunction!(dm, d1, d2)

        layer = DisjunctiveProjectionLayer(
            dm;
            y_regularization = 1e-4,
            ycopy_regularization = 1e-4,
            gamma_regularization = 1e-4,
            anchor_regularization = 1e-4,
        )

        model = Chain(
            Dense(2 => 8, relu),
            Dense(8 => 2),
            layer,
        )

        x = Float32[0.3, 0.4]

        y = model(x)

        @test length(y) == 2
        @test y[1] >= -1e-6
        @test y[1] <= 1.0 + 1e-6
        @test y[2] >= -1e-6
        @test y[2] <= 1.0 + 1e-6
        @test y[1] + y[2] >= 0.8 - 1e-6

        ps = Flux.trainables(model)
        @test !isempty(ps)

        loss(m, x) = sum(m(x))
        grads = Flux.gradient(m -> loss(m, x), model)
        @test grads !== nothing
    end

    @testset "Differentiation stress: symmetric 4-scenario hull" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        # Global equality: keeps the solution on a line.
        add_linear_constraint!(dm, [1.0, 1.0], :(==), 1.0)

        # Disjunction 1:
        # y1 <= 0.25 OR y1 >= 0.75
        add_disjunction!(
            dm,
            [LinearConstraint([1.0, 0.0], :<=, 0.25)],
            [LinearConstraint([1.0, 0.0], :>=, 0.75)],
        )

        # Disjunction 2:
        # y2 <= 0.25 OR y2 >= 0.75
        add_disjunction!(
            dm,
            [LinearConstraint([0.0, 1.0], :<=, 0.25)],
            [LinearConstraint([0.0, 1.0], :>=, 0.75)],
        )

        layer = DisjunctiveProjectionLayer(
            dm;
            y_regularization = 1e-4,
            ycopy_regularization = 1e-4,
            gamma_regularization = 1e-4,
            anchor_regularization = 1e-4,
        )

        # Exactly central. This is intentionally degenerate.
        yhat = [0.5, 0.5]

        y = layer(yhat)

        @test length(y) == 2
        @test all(isfinite, y)
        @test isapprox(sum(y), 1.0; atol = 1e-5)

        grad = Zygote.gradient(z -> sum(layer(z)), yhat)[1]

        @test grad !== nothing
        @test length(grad) == 2
        @test all(isfinite, grad)
    end

    @testset "Differentiation stress: redundant constraints" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        # Same equality expressed redundantly.
        add_linear_constraint!(dm, [1.0, 1.0], :(==), 1.0)
        add_linear_constraint!(dm, [2.0, 2.0], :(==), 2.0)

        # Redundant inequalities implied by the equality.
        add_linear_constraint!(dm, [1.0, 1.0], :>=, 1.0)
        add_linear_constraint!(dm, [1.0, 1.0], :<=, 1.0)

        add_disjunction!(
            dm,
            [LinearConstraint([1.0, 0.0], :<=, 0.25)],
            [LinearConstraint([1.0, 0.0], :>=, 0.75)],
        )

        add_disjunction!(
            dm,
            [LinearConstraint([0.0, 1.0], :<=, 0.25)],
            [LinearConstraint([0.0, 1.0], :>=, 0.75)],
        )

        layer = DisjunctiveProjectionLayer(
            dm;
            anchor_regularization = 1e-3,
        )

        yhat = [0.5, 0.5]

        y = layer(yhat)

        @test length(y) == 2
        @test all(isfinite, y)
        @test isapprox(sum(y), 1.0; atol = 1e-5)

        grad = Zygote.gradient(z -> sum(layer(z)), yhat)[1]

        @test grad !== nothing
        @test length(grad) == 2
        @test all(isfinite, grad)
    end

    @testset "Differentiation stress: 3D 8-scenario hull" begin
        dm = DisjunctiveModel(3)

        set_bounds!(
            dm,
            lower = [0.0, 0.0, 0.0],
            upper = [1.0, 1.0, 1.0],
        )

        # Global simplex equality.
        add_linear_constraint!(dm, [1.0, 1.0, 1.0], :(==), 1.0)

        # Symmetric split on each variable.
        add_disjunction!(
            dm,
            [LinearConstraint([1.0, 0.0, 0.0], :<=, 0.2)],
            [LinearConstraint([1.0, 0.0, 0.0], :>=, 0.6)],
        )

        add_disjunction!(
            dm,
            [LinearConstraint([0.0, 1.0, 0.0], :<=, 0.2)],
            [LinearConstraint([0.0, 1.0, 0.0], :>=, 0.6)],
        )

        add_disjunction!(
            dm,
            [LinearConstraint([0.0, 0.0, 1.0], :<=, 0.2)],
            [LinearConstraint([0.0, 0.0, 1.0], :>=, 0.6)],
        )

        layer = DisjunctiveProjectionLayer(
            dm;
            anchor_regularization = 1e-3,
        )

        yhat = [1/3, 1/3, 1/3]

        y = layer(yhat)

        @test length(y) == 3
        @test all(isfinite, y)
        @test isapprox(sum(y), 1.0; atol = 1e-5)

        grad = Zygote.gradient(z -> sum(layer(z)), yhat)[1]

        @test grad !== nothing
        @test length(grad) == 3
        @test all(isfinite, grad)
    end

    @testset "DisjunctiveProjectionLayer" begin
        dm = DisjunctiveModel(3)

        layer = DisjunctiveProjectionLayer(
            dm;
            y_regularization = 0.0,
            ycopy_regularization = 0.0,
            gamma_regularization = 0.0,
            anchor_regularization = 0.0
        )
        @test projection_mode(layer) == :dnf_qp

        milp_layer = DisjunctiveProjectionLayer(dm; mode = :milp)
        @test projection_mode(milp_layer) == :milp
        @test milp_layer.config.gradient == :straight_through

        yhat = [1.0, 2.0, 3.0]
        @test isapprox(layer(yhat), yhat; atol = 1e-3)
    end

    @testset "CNF and DNF formulation selection" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        add_disjunction!(
            dm,
            [LinearConstraint([1.0, 0.0], :<=, 0.25)],
            [LinearConstraint([1.0, 0.0], :>=, 0.75)],
        )

        add_disjunction!(
            dm,
            [LinearConstraint([0.0, 1.0], :<=, 0.25)],
            [LinearConstraint([0.0, 1.0], :>=, 0.75)],
        )

        dnf = convex_hull_form(dm; prune_infeasible = false)
        cnf = cnf_hull_form(dm)

        @test num_scenarios(dnf) == 4
        @test num_blocks(cnf) == 2
        @test length(cnf.blocks[1].disjuncts) == 2
        @test length(cnf.blocks[2].disjuncts) == 2

        yhat = [0.5, 0.5]

        dnf_layer = DisjunctiveProjectionLayer(dm; formulation = :dnf)
        cnf_layer = DisjunctiveProjectionLayer(dm; formulation = :cnf)

        y_dnf = dnf_layer(yhat)
        y_cnf = cnf_layer(yhat)

        @test length(y_dnf) == 2
        @test length(y_cnf) == 2
        @test all(isfinite, y_dnf)
        @test all(isfinite, y_cnf)

        @test projection_formulation(dnf_layer) == :dnf
        @test projection_formulation(cnf_layer) == :cnf
    end

    @testset "Display utilities" begin
        dm = DisjunctiveModel(2)
        set_bounds!(dm, lower = [0.0, 0.0], upper = [1.0, 1.0])

        add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

        add_disjunction!(
            dm,
            [LinearConstraint([1.0, 0.0], :<=, 0.25)],
            [LinearConstraint([1.0, 0.0], :>=, 0.75)],
        )

        buf = IOBuffer()
        print_model(dm; io = buf)
        str = String(take!(buf))

        @test occursin("DisjunctiveModel", str)
        @test occursin("global constraints", str)
        @test occursin("disjunction[1]", str)

        buf = IOBuffer()
        print_projection_model(dm; formulation = :dnf, io = buf)
        str = String(take!(buf))

        @test occursin("DNF ConvexHullForm", str)

        buf = IOBuffer()
        print_projection_model(dm; formulation = :cnf, io = buf)
        str = String(take!(buf))

        @test occursin("CNF ConvexHullForm", str)
    end

    @testset "CNF projection backend" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        add_disjunction!(
            dm,
            [LinearConstraint([1.0, 0.0], :<=, 0.25)],
            [LinearConstraint([1.0, 0.0], :>=, 0.75)],
        )

        add_disjunction!(
            dm,
            [LinearConstraint([0.0, 1.0], :<=, 0.25)],
            [LinearConstraint([0.0, 1.0], :>=, 0.75)],
        )

        yhat = [0.5, 0.5]

        result = project(dm, yhat; formulation = :cnf)

        @test result.status == MOI.OPTIMAL
        @test length(result.y) == 2
        @test all(isfinite, result.y)
        @test result.y[1] >= -1e-6
        @test result.y[1] <= 1.0 + 1e-6
        @test result.y[2] >= -1e-6
        @test result.y[2] <= 1.0 + 1e-6

        layer = DisjunctiveProjectionLayer(dm; formulation = :cnf)
        y = layer(yhat)

        @test length(y) == 2
        @test all(isfinite, y)
    end


end