using Test
using DisjunctiveDifferentiableLayers

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

        hull = convex_hull_form(sm)

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

    @testset "DisjunctiveProjectionLayer" begin
        dm = DisjunctiveModel(3)

        layer = DisjunctiveProjectionLayer(dm)
        @test projection_mode(layer) == :dnf_qp

        milp_layer = DisjunctiveProjectionLayer(dm; mode = :milp)
        @test projection_mode(milp_layer) == :milp
        @test milp_layer.config.gradient == :straight_through

        yhat = [1.0, 2.0, 3.0]
        @test layer(yhat) == yhat
    end
end