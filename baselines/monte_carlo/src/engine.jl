# AMS selection/replay/weight logic ported from the existing PEAS shell engine.
mutable struct AMSModReplica
    x::Vector{Float64}
    step::Int
    rng::Xoshiro
    noise::Vector{Float64}

    # Smaller values mean greater progress toward the receiver.
    max_progress_val::Float64
    max_progress_pos::Vector{Float64}
    max_progress_step::Int

    # State and RNG immediately after the most recently crossed AMS level.
    restart_pos::Vector{Float64}
    restart_step::Int
    restart_rng::Xoshiro
end

function AMSModReplica(x::Vector{Float64}, rng::Xoshiro)
    return AMSModReplica(
        copy(x),
        0,
        rng,
        zeros(3),
        Inf,
        copy(x),
        0,
        copy(x),
        0,
        copy(rng),
    )
end

@inline function update_max_progress!(rep::AMSModReplica, value::Float64)
    if value <= rep.max_progress_val
        rep.max_progress_val = value
        rep.max_progress_step = rep.step
        rep.max_progress_pos .= rep.x
    end
    return nothing
end

function diffusion_path!(
    rep::AMSModReplica,
    domain::AMSModDomain,
    sigma::Float64,
    num_steps::Int,
)
    for _ in (rep.step + 1):num_steps
        status = advance_one_step!(domain, rep, sigma)
        # A Robin-absorbed proposal lies beyond the valid domain. It is a
        # terminal failure and must not become a replay/cloning checkpoint.
        status != :failed && update_max_progress!(rep, progress_squared(domain, rep.x))
        status == :absorbed && return true
        status == :failed && return false
    end
    return false
end

@inline function save_restart!(rep::AMSModReplica)
    rep.restart_pos .= rep.x
    rep.restart_step = rep.step
    return nothing
end

@inline function find_first_crossing!(
    rep::AMSModReplica,
    domain::AMSModDomain,
    sigma::Float64,
    level_current::Float64,
    num_steps::Int,
)
    current = progress_squared(domain, rep.x)
    update_max_progress!(rep, current)
    if receiver_hit(domain, rep.x)
        save_restart!(rep)
        return :absorbed
    end
    if current <= level_current
        save_restart!(rep)
        return :crossed
    end

    for _ in (rep.step + 1):num_steps
        status = advance_one_step!(domain, rep, sigma)
        status == :failed && return :failed
        current = progress_squared(domain, rep.x)
        update_max_progress!(rep, current)

        if status == :absorbed
            save_restart!(rep)
            return :absorbed
        end
        if current <= level_current
            save_restart!(rep)
            return :crossed
        end
    end
    return :failed
end

function clone_replica!(
    child::AMSModReplica,
    new_rng::Xoshiro,
    domain::AMSModDomain,
    sigma::Float64,
    level_current::Float64,
    num_steps::Int,
)
    status = find_first_crossing!(child, domain, sigma, level_current, num_steps)
    status == :failed && error("selected parent did not cross current level")
    status == :absorbed && return true

    child.rng = new_rng
    child.restart_rng = copy(new_rng)
    return diffusion_path!(child, domain, sigma, num_steps)
end

function select_worst_survivors!(
    worst_indices::Vector{Int},
    worst_vals::Vector{Float64},
    survivor_indices::Vector{Int},
    killed::BitVector,
    replicas::Vector{AMSModReplica},
    n_worst::Int,
)
    fill!(killed, false)
    fill!(worst_indices, 0)
    fill!(worst_vals, -Inf)

    # This intentionally mirrors canonical/ams_mod.jl instead of sorting all
    # replicas, preserving its level and tie-selection semantics.
    for i in eachindex(replicas)
        value = replicas[i].max_progress_val
        min_pos = 1
        min_value = worst_vals[1]
        for j in 2:n_worst
            if worst_vals[j] < min_value
                min_pos = j
                min_value = worst_vals[j]
            end
        end
        if value > min_value
            worst_vals[min_pos] = value
            worst_indices[min_pos] = i
        end
    end

    level_current = minimum(worst_vals)
    for index in worst_indices
        killed[index] = true
    end

    n_survivor = 0
    for i in eachindex(replicas)
        if !killed[i]
            n_survivor += 1
            survivor_indices[n_survivor] = i
        end
    end
    return level_current, n_survivor
end

function tau_distribution(replicas::Vector{AMSModReplica}, hits::BitVector, dt::Float64, t_max::Float64)
    tau = Float64[
        min(replicas[i].step * dt, t_max)
        for i in eachindex(replicas)
        if hits[i]
    ]
    sort!(tau)
    return tau
end

function absorption_curve!(q::AbstractVector{Float64}, q_hat::Float64, tau::Vector{Float64}, times::Vector{Float64})
    fill!(q, 0.0)
    isempty(tau) && return q
    sort!(tau)
    tau_index = 1
    absorbed = 0
    for i in eachindex(times)
        while tau_index <= length(tau) && tau[tau_index] <= times[i]
            absorbed += 1
            tau_index += 1
        end
        q[i] = q_hat * absorbed / length(tau)
    end
    return q
end

function run_ams_mod!(
    replicas::Vector{AMSModReplica},
    domain::AMSModDomain,
    sigma::Float64,
    num_steps::Int,
    repeat_seed::Int;
    seed_policy::Symbol=:stable,
    n_worst::Int=1,
    max_iterations::Int=typemax(Int),
    dt::Float64=1.0,
    t_max::Float64=num_steps * dt,
    times::Vector{Float64}=collect(0:num_steps) .* dt,
)
    num_replicas = length(replicas)
    1 <= n_worst < num_replicas || error("n_worst must satisfy 1 <= n_worst < num_replicas.")
    max_iterations >= 0 || error("max_iterations must be nonnegative.")

    hits = falses(num_replicas)
    rng_clone = Xoshiro(seed_for(seed_policy, repeat_seed, :clone))
    worst_indices = Vector{Int}(undef, n_worst)
    worst_vals = Vector{Float64}(undef, n_worst)
    survivor_indices = Vector{Int}(undef, num_replicas - n_worst)
    killed = falses(num_replicas)
    num_replica_total = num_replicas

    # Canonical first round: every replica evolves independently to the full
    # time horizon or a terminal boundary event.
    for i in eachindex(replicas)
        replicas[i].restart_rng = copy(replicas[i].rng)
        hits[i] = diffusion_path!(replicas[i], domain, sigma, num_steps)
    end

    receiver_level = target_progress(domain)
    level_current = Inf
    iteration = 0
    while level_current > receiver_level && iteration < max_iterations
        all(hits) && break
        level_current, n_survivor = select_worst_survivors!(
            worst_indices,
            worst_vals,
            survivor_indices,
            killed,
            replicas,
            n_worst,
        )
        level_current <= receiver_level && break
        iteration += 1

        for index in worst_indices
            parent_index = survivor_indices[rand(rng_clone, 1:n_survivor)]
            parent = replicas[parent_index]
            child_start = copy(parent.restart_pos)
            child_start_progress = progress_squared(domain, child_start)
            child = AMSModReplica(
                child_start,
                parent.restart_step,
                copy(parent.restart_rng),
                zeros(3),
                child_start_progress,
                copy(child_start),
                parent.restart_step,
                copy(parent.restart_pos),
                parent.restart_step,
                copy(parent.restart_rng),
            )
            new_rng = Xoshiro(seed_for(seed_policy, repeat_seed, :child, iteration, index, num_replica_total))
            hits[index] = clone_replica!(
                child,
                new_rng,
                domain,
                sigma,
                level_current,
                num_steps,
            )
            replicas[index] = child
            num_replica_total += 1
        end
    end

    tau = tau_distribution(replicas, hits, dt, t_max)
    q_hat = ((num_replicas - n_worst) / num_replicas)^iteration * count(hits) / num_replicas
    q = zeros(Float64, length(times))
    absorption_curve!(q, q_hat, tau, times)
    return (
        q=q,
        q_hat=q_hat,
        tau=tau,
        hits=hits,
        level_current=level_current,
        iterations=iteration,
        num_replica_total=num_replica_total,
        num_final_success=count(hits),
        converged=all(hits) || level_current <= receiver_level,
    )
end


"""One independent AMS repetition; finite-step RX endpoint-hit estimator."""
function run_single(; domain::AMSModDomain,x0::Vector{Float64},dt::Float64,t_max::Float64,
                    num_replicas::Int,n_worst::Int,seed::Int,times=collect(0:round(Int,t_max/dt)).*dt,
                    max_iterations::Int=100000,seed_policy::Symbol=:stable)
    validate_domain(domain,x0)
    dt>0 && t_max>0 || error("dt and t_max must be positive")
    steps=round(Int,t_max/dt)
    isapprox(steps*dt,t_max;atol=1e-10,rtol=1e-12) || error("t_max/dt must be integer")
    num_replicas>=2 || error("At least 2 replicas required")
    issorted(times) && first(times)>=0 && last(times)<=t_max+1e-10 || error("Invalid query times")
    replicas=[AMSModReplica(copy(x0),Xoshiro(seed_for(seed_policy,seed,:replica,i))) for i in 1:num_replicas]
    return run_ams_mod!(replicas,domain,sqrt(2*dt),steps,seed;n_worst=n_worst,
                       max_iterations=max_iterations,dt=dt,t_max=t_max,times=Float64.(times),seed_policy=seed_policy)
end
