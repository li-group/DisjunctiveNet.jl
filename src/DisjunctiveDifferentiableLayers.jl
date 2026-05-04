module DisjunctiveDifferentiableLayers

include("types.jl")
include("utils.jl")
include("modeling.jl")
include("projection_layer.jl")

export DisjunctiveModel
export DisjunctiveProjectionLayer
export ProjectionConfig

export LinearConstraint
export Disjunction
export output_dimension
export lower_bounds
export upper_bounds
export set_bounds!
export add_linear_constraint!
export add_disjunction!
export linear_constraints
export disjunctions
export projection_mode

end