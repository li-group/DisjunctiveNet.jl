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