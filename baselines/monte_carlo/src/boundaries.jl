function outer_absorb_probability(kappa::Float64, dt::Float64)
    kappa <= 0.0 && return 0.0
    return clamp(kappa * sqrt(pi * dt), 0.0, 1.0)
end

function reflect_outer(
    p1::Vector{Float64},
    p2::Vector{Float64},
    g::ShellGeometry,
    rng::AbstractRNG,
    p_absorb::Float64;
    max_bounces::Int=8,
)
    c = g.outer_center
    r = g.outer_radius
    r2 = r * r
    px = copy(p2)
    start = copy(p1)
    nb = 0
    while nb < max_bounces && sum(abs2, px - c) > r2
        d = px - start
        a = dot(d, d)
        a == 0.0 && break
        m = start - c
        b = 2.0 * dot(m, d)
        cc = dot(m, m) - r2
        disc = b * b - 4.0 * a * cc
        disc < 0.0 && break
        sqrt_disc = sqrt(max(disc, 0.0))
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        t = Inf
        0.0 <= t1 <= 1.0 && (t = min(t, t1))
        0.0 <= t2 <= 1.0 && (t = min(t, t2))
        isfinite(t) || break

        if p_absorb > 0.0 && rand(rng) < p_absorb
            return px, true
        end

        hit = start + t .* d
        normal = (hit - c) ./ r
        rem = px - hit
        px = hit + rem - 2.0 * dot(rem, normal) .* normal
        start = hit - 1.0e-9 .* normal
        nb += 1
    end
    return px, false
end

function radial_hit_time(start::Vector{Float64}, target::Vector{Float64}, radius::Float64)
    d = target - start
    a = d[1]^2 + d[2]^2
    a == 0.0 && return Inf
    b = 2.0 * (start[1] * d[1] + start[2] * d[2])
    c = start[1]^2 + start[2]^2 - radius^2
    disc = b^2 - 4.0 * a * c
    disc < 0.0 && return Inf
    sqrt_disc = sqrt(max(disc, 0.0))
    roots = ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a))
    ts = [t for t in roots if 0.0 <= t <= 1.0]
    return isempty(ts) ? Inf : minimum(ts)
end

function cap_hit_time(start::Vector{Float64}, target::Vector{Float64}, g::PipeGeometry)
    dz = target[3] - start[3]
    dz == 0.0 && return Inf, 0.0
    cap_z = target[3] > g.z_max ? g.z_max : target[3] < g.z_min ? g.z_min : NaN
    isnan(cap_z) && return Inf, 0.0
    t = (cap_z - start[3]) / dz
    return 0.0 <= t <= 1.0 ? (t, cap_z) : (Inf, 0.0)
end

function reflect_pipe_step(
    p1::Vector{Float64},
    p2::Vector{Float64},
    g::PipeGeometry,
    rng::AbstractRNG,
    p_wall_absorb::Float64;
    max_bounces::Int=12,
)
    px = copy(p2)
    start = copy(p1)
    for _ in 1:max_bounces
        outside_wall = radial_norm_xy(px) > g.radius
        outside_caps = px[3] < g.z_min || px[3] > g.z_max
        (!outside_wall && !outside_caps) && return px, false

        t_wall = outside_wall ? radial_hit_time(start, px, g.radius) : Inf
        t_cap, cap_z = outside_caps ? cap_hit_time(start, px, g) : (Inf, 0.0)
        if t_cap <= t_wall
            if g.end_boundary == "absorbing"
                return px, true
            end
            hit = start + t_cap .* (px - start)
            rem = px - hit
            rem[3] = -rem[3]
            px = hit + rem
            start = copy(hit)
            start[3] += cap_z == g.z_max ? -1.0e-9 : 1.0e-9
        else
            if p_wall_absorb > 0.0 && rand(rng) < p_wall_absorb
                return px, true
            end
            hit = start + t_wall .* (px - start)
            normal = Float64[hit[1], hit[2], 0.0] ./ max(radial_norm_xy(hit), 1.0e-12)
            rem = px - hit
            px = hit + rem - 2.0 * dot(rem, normal) .* normal
            start = hit - 1.0e-9 .* normal
        end
    end
    return px, !inside_pipe(px, g)
end


# The Brownian draw uses a reusable 3-vector; boundary buffers are allocated
# only on actual wall crossings. This preserves the legacy draw order.
@inline function advance_one_step!(domain::AMSModDomain,rep::AMSModReplica,sigma::Float64)
    x=rep.x
    a,b,c=x[1],x[2],x[3]
    randn!(rep.rng,rep.noise)
    x[1]=a+sigma*rep.noise[1]
    x[2]=b+sigma*rep.noise[2]
    x[3]=c+sigma*rep.noise[3]
    rep.step+=1
    receiver_hit(domain,x) && return :absorbed
    if domain isa FiniteShellRobinDomain
        g=domain.geometry
        if (x[1]-g.outer_center[1])^2+(x[2]-g.outer_center[2])^2+(x[3]-g.outer_center[3])^2>g.outer_radius^2
            proposed,lost=reflect_outer([a,b,c],x,g,rep.rng,domain.outer_p_absorb)
            x.=proposed
            lost && return :failed
            norm(x-g.outer_center)<=g.outer_radius+1e-10 || error("Sphere reflection exceeded bounce limit")
        end
    elseif domain isa FinitePipeRobinDomain
        g=domain.geometry
        if !inside_pipe(x,g)
            proposed,lost=reflect_pipe_step([a,b,c],x,g,rep.rng,domain.wall_p_absorb)
            x.=proposed
            lost && return :failed
        end
    end
    return :active
end
