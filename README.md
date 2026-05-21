# DisjunctiveDifferentiableLayers.jl

`DisjunctiveDifferentiableLayers.jl` provides differentiable projection layers for neural networks whose outputs must satisfy mixed logical-linear constraints.

The package lets users define rules of the form

```text
global linear constraints
+
(disjunct 1 OR disjunct 2 OR ...)
+
(disjunct 1 OR disjunct 2 OR ...)
+ ...
```

and automatically builds a differentiable projection layer that can be placed after a Flux neural network.

The layer supports three lifted formulations:

| Formulation | Description |
|---|---|
| `:dnf` | Full DNF convex-hull formulation. Stronger but can grow combinatorially. |
| `:cnf` | One convex-hull block per disjunction. Scalable but generally weaker. |
| `:partial_dnf` | Expands a chosen subset of rules into DNF and keeps the rest as CNF. |

The package is designed for research on differentiable constrained learning, neural-symbolic learning, and safe neural network prediction.

---

## Installation

This package currently uses a modified version of `DiffOpt.jl` with a regularized KKT solve for more stable reverse differentiation.

Until the DiffOpt patch is upstreamed, install the modified DiffOpt first:

```julia
using Pkg

Pkg.add(url = "https://github.com/Shraman_Pal/DiffOpt.jl")
Pkg.add(url = "https://github.com/YOUR_USERNAME/DisjunctiveDifferentiableLayers.jl")
```

For local development:

```bash
git clone https://github.com/YOUR_USERNAME/DisjunctiveDifferentiableLayers.jl
cd DisjunctiveDifferentiableLayers.jl
julia --project=.
```

Then:

```julia
using Pkg
Pkg.instantiate()
Pkg.test()
```

---

## Basic idea

A neural network produces an unconstrained prediction:

```julia
yhat = backbone(x)
```

The projection layer maps it to a feasible point:

```julia
y = projection_layer(yhat)
```

The full constrained model is:

```julia
x -> backbone(x) -> disjunctive projection layer -> feasible y
```

The projection is differentiable, so training can be performed using ordinary Flux/Zygote code.

---

## A first disjunctive model

Suppose the neural network predicts two outputs:

```text
y[1], y[2]
```

We want:

```text
0 <= y[1] <= 1
0 <= y[2] <= 1

global rule:
y[1] + y[2] >= 0.8

rule 1:
y[1] <= 0.25 OR y[1] >= 0.75

rule 2:
y[2] <= 0.25 OR y[2] >= 0.75
```

This can be modeled as:

```julia
using DisjunctiveDifferentiableLayers

dm = DisjunctiveModel(2)

set_bounds!(
    dm,
    lower = [0.0, 0.0],
    upper = [1.0, 1.0],
)

add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

add_disjunction!(
    dm,
    [LinearConstraint([1.0, 0.0], :<=, 0.25)],
    [LinearConstraint([1.0, 0.0], :>=, 0.75)];
    name = :x_split,
)

add_disjunction!(
    dm,
    [LinearConstraint([0.0, 1.0], :<=, 0.25)],
    [LinearConstraint([0.0, 1.0], :>=, 0.75)];
    name = :y_split,
)
```

Inspect the model:

```julia
print_model(dm)
```

Inspect the lifted projection formulation:

```julia
print_projection_model(dm; formulation = :dnf)
print_projection_model(dm; formulation = :cnf)
print_projection_model(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
)
```

---

## Projection API

Project a raw prediction:

```julia
yhat = [0.5, 0.1]

result = project(dm, yhat; formulation = :dnf)

println(result.status)
println(result.y)
```

Use a projection layer directly:

```julia
layer = DisjunctiveProjectionLayer(dm; formulation = :dnf)

y = layer(yhat)
```

Available formulations:

```julia
layer_dnf = DisjunctiveProjectionLayer(dm; formulation = :dnf)

layer_cnf = DisjunctiveProjectionLayer(dm; formulation = :cnf)

layer_partial = DisjunctiveProjectionLayer(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
)
```

---

## Full Flux example

The following example builds a small neural network, adds a disjunctive projection layer, runs one training step, and performs inference on a single sample.

```julia
using Flux
using Zygote
using DisjunctiveDifferentiableLayers

# -----------------------------
# 1. Build a neural network
# -----------------------------

backbone = Chain(
    Dense(3 => 8, relu),
    Dense(8 => 2),
)

x = Float32[0.2, 0.7, 0.4]
target = Float32[0.8, 0.2]

# -----------------------------
# 2. Build the constrained model
# -----------------------------

model = constrained_model(
    backbone,
    x;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_rule, :y_rule],
    y_regularization = 1e-4,
    ycopy_regularization = 1e-4,
    gamma_regularization = 1e-4,
    anchor_regularization = 1e-4,
) do dm

    set_bounds!(
        dm,
        lower = [0.0, 0.0],
        upper = [1.0, 1.0],
    )

    # Global constraint:
    # y1 + y2 >= 0.8
    add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

    # Rule 1 has three disjuncts:
    # y1 <= 0.2 OR 0.4 <= y1 <= 0.6 OR y1 >= 0.8
    add_disjunction!(
        dm,
        [LinearConstraint([1.0, 0.0], :<=, 0.2)],
        [
            LinearConstraint([1.0, 0.0], :>=, 0.4),
            LinearConstraint([1.0, 0.0], :<=, 0.6),
        ],
        [LinearConstraint([1.0, 0.0], :>=, 0.8)];
        name = :x_rule,
    )

    # Rule 2 has three disjuncts:
    # y2 <= 0.2 OR 0.35 <= y2 <= 0.55 OR y2 >= 0.7
    add_disjunction!(
        dm,
        [LinearConstraint([0.0, 1.0], :<=, 0.2)],
        [
            LinearConstraint([0.0, 1.0], :>=, 0.35),
            LinearConstraint([0.0, 1.0], :<=, 0.55),
        ],
        [LinearConstraint([0.0, 1.0], :>=, 0.7)];
        name = :y_rule,
    )
end

# -----------------------------
# 3. Inspect the projection layer
# -----------------------------

print_projection_model(
    model.layer.model;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_rule, :y_rule],
)

# -----------------------------
# 4. Forward pass
# -----------------------------

y = model(x)

println("Projected prediction = ", y)
println("Feasibility check: y1 + y2 = ", sum(y))

# -----------------------------
# 5. One training step
# -----------------------------

loss(m, x, target) = sum(abs2, m(x) .- target)

opt = Flux.setup(Adam(1e-3), model)

l, grads = Flux.withgradient(model) do m
    loss(m, x, target)
end

Flux.update!(opt, model, grads[1])

println("Training loss before update = ", l)

# -----------------------------
# 6. Inference after one update
# -----------------------------

y_after = model(x)

println("Projected prediction after one update = ", y_after)
println("Feasibility check after update: y1 + y2 = ", sum(y_after))
```

The user never has to write an adjoint or reverse pass. The constrained model is a Flux-compatible layer.

---

## Formulation summaries and benchmarking

The package includes utilities for inspecting lifted formulations:

```julia
formulation_summary(dm; formulation = :dnf)

formulation_summary(dm; formulation = :cnf)

formulation_summary(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
)
```

Benchmark projection time and model size:

```julia
benchmark_projection(dm, yhat; formulation = :cnf, label = "CNF")

benchmark_projection(
    dm,
    yhat;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
    label = "partial-DNF k=1",
)
```

Example output:

```text
CNF                  status=OPTIMAL build=0.002s solve=0.003s vars=54 cons=152 y=[0.5, 0.2]
partial-DNF k=1      status=OPTIMAL build=0.003s solve=0.004s vars=62 cons=180 y=[0.5, 0.2]
```

---

## Examples

Run examples from the package root:

```bash
julia --project=. examples/01_basic_dnf_cnf_partial.jl
julia --project=. examples/02_comprehensive_formulations.jl
julia --project=. examples/03_large_scale_comparison.jl
```

---

## Current limitations

- Constraints are currently linear.
- The differentiable backend relies on DiffOpt.
- A modified DiffOpt with regularized KKT solves is recommended for robust reverse differentiation.
- Full DNF can grow exponentially in the number of disjunctions.
- Partial-DNF is intended to trade off formulation strength and computational size.

---

## Roadmap

Planned improvements:

- JuMP-style constraint input, so users can write `y[1] + y[2] <= 1.0`.
- Better rule ordering heuristics for partial-DNF.
- MILP baseline projection mode.
- Upstreaming the regularized KKT solve into DiffOpt.
- More examples and benchmark scripts.