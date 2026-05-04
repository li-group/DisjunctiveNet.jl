module DisjunctiveDifferentiableLayers

include("types.jl")
include("utils.jl")
include("modeling.jl")
include("standard_form.jl")
include("convex_hull.jl")
include("projection_backend.jl")
include("differentiation.jl")
include("projection_layer.jl")

export DisjunctiveModel
export DisjunctiveProjectionLayer
export ProjectionConfig

export LinearConstraint
export Disjunction
export StandardDisjunctiveModel
export ConvexHullForm
export ConvexHullScenario

export output_dimension
export lower_bounds
export upper_bounds
export set_bounds!
export add_linear_constraint!
export add_disjunction!
export linear_constraints
export disjunctions

export standard_form
export convex_hull_form
export num_scenarios

export projection_mode

export ProjectionResult
export project
export build_projection_model

end