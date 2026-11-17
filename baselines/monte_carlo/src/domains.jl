# Dimensionless geometries. D=1; RX radius=1 in the supplied cases.
abstract type AMSModDomain end
struct UnboundedDomain <: AMSModDomain
    rx_center::Vector{Float64}
    rx_radius::Float64
end
struct ShellGeometry
    rx_center::Vector{Float64}
    outer_center::Vector{Float64}
    rx_radius::Float64
    outer_radius::Float64
    kappa::Float64
    use_outer_boundary::Bool
end
struct PipeGeometry
    radius::Float64
    z_min::Float64
    z_max::Float64
    wall_kappa::Float64
    end_boundary::String
    rx_center::Vector{Float64}
    rx_radius::Float64
end
struct FiniteShellRobinDomain <: AMSModDomain
    geometry::ShellGeometry
    outer_p_absorb::Float64
end
struct FinitePipeRobinDomain <: AMSModDomain
    geometry::PipeGeometry
    wall_p_absorb::Float64
end
rx_center(d::UnboundedDomain) = d.rx_center
rx_center(d::Union{FiniteShellRobinDomain,FinitePipeRobinDomain}) = d.geometry.rx_center
rx_radius(d::UnboundedDomain) = d.rx_radius
rx_radius(d::Union{FiniteShellRobinDomain,FinitePipeRobinDomain}) = d.geometry.rx_radius
@inline function progress_squared(domain::AMSModDomain, x::Vector{Float64})
    c=rx_center(domain)
    return (x[1]-c[1])^2+(x[2]-c[2])^2+(x[3]-c[3])^2
end
@inline receiver_hit(d::AMSModDomain,x::Vector{Float64}) = progress_squared(d,x)<=rx_radius(d)^2
@inline target_progress(d::AMSModDomain) = rx_radius(d)^2
radial_norm_xy(x::Vector{Float64}) = hypot(x[1],x[2])
inside_pipe(x::Vector{Float64},g::PipeGeometry; margin=0.0) = radial_norm_xy(x)<=g.radius-margin && g.z_min+margin<=x[3]<=g.z_max-margin
function validate_domain(d::AMSModDomain,x0::Vector{Float64})
    length(x0)==3 && all(isfinite,x0) || error("Expected finite Cartesian 3-vector")
    progress_squared(d,x0)>rx_radius(d)^2 || error("TX is inside or on RX: $x0")
    if d isa FiniteShellRobinDomain
        g=d.geometry
        norm(x0-g.outer_center)<g.outer_radius || error("TX is outside or on outer sphere")
    elseif d isa FinitePipeRobinDomain
        inside_pipe(x0,d.geometry; margin=1e-12) || error("TX is outside or on pipe boundary")
    end
    return nothing
end
