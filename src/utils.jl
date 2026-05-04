function _validate_projection_config(config::ProjectionConfig)
    valid_modes = (:dnf_qp, :milp)
    valid_gradients = (:diffopt, :straight_through)
    valid_fallbacks = (:identity, :error)

    config.mode in valid_modes ||
        throw(ArgumentError("Invalid projection mode $(config.mode). Expected one of $(valid_modes)."))

    config.gradient in valid_gradients ||
        throw(ArgumentError("Invalid gradient mode $(config.gradient). Expected one of $(valid_gradients)."))

    config.fallback in valid_fallbacks ||
        throw(ArgumentError("Invalid fallback $(config.fallback). Expected one of $(valid_fallbacks)."))

    if config.mode == :milp && config.gradient == :diffopt
        throw(ArgumentError("MILP mode does not support `gradient = :diffopt`; use `gradient = :straight_through`."))
    end

    if config.mode == :dnf_qp && config.gradient == :straight_through
        # This is allowed, but not the intended default.
        return nothing
    end

    return nothing
end