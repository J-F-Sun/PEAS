const SCENARIOS=("canonical","concentric","eccentric","pipe")
vec3(g,k)=Float64.(g[k])
function load_config(path::AbstractString)
    path=abspath(path)
    c=TOML.parsefile(path)
    c["schema_version"]==1 || error("Unsupported schema")
    c["scenario"] in SCENARIOS || error("Unknown scenario")
    g=c["geometry"]
    length(g["rx_center"])==3 && all(isfinite,g["rx_center"]) || error("Invalid RX center")
    isfinite(g["rx_radius"]) && g["rx_radius"]>0 || error("Invalid RX radius")
    c["outputs"]["checkpoint_batch"]>=1 || error("Invalid checkpoint batch")
    ds=c["dataset"]
    ds["min_distance"]>g["rx_radius"] && ds["max_distance"]>ds["min_distance"] || error("Invalid distance interval")
    all(k->ds[k]>0,("train","validation","test")) || error("Invalid split sizes")
    ds["boundary_margin"]>=0 || error("Negative boundary margin")
    # Paths always resolve relative to the config file, even when invoked elsewhere.
    ds["manifest"]=normpath(joinpath(dirname(path),ds["manifest"]))
    c["outputs"]["root"]=abspath(get(ENV,"AMS_OUTPUT_ROOT",joinpath(dirname(path),c["outputs"]["root"])))
    c["config_path"]=path
    settings(c,"formal"); settings(c,"smoke")
    build_domain(c,Float64(c["simulation"]["dt"]))
    return c
end
function settings(c,profile)
    profile in ("formal","smoke") || error("profile must be formal or smoke")
    s=deepcopy(c["simulation"])
    profile=="smoke" && merge!(s,c["smoke"])
    s["dt"]>0 && s["t_max"]>0 || error("Invalid time interval")
    n=round(Int,s["t_max"]/s["dt"])
    isapprox(n*s["dt"],s["t_max"];atol=1e-10,rtol=1e-12) || error("t_max/dt must be integer")
    1<=s["n_worst"]<s["num_replicas"] || error("Invalid replica/kill count")
    s["repeats"]>=2 && s["output_points"]>=2 && s["max_iterations"]>0 || error("Invalid budget")
    return s
end
output_times(s)=collect(range(0.0,Float64(s["t_max"]);length=Int(s["output_points"])))
function build_domain(c,dt)
    scenario=c["scenario"]; g=c["geometry"]
    rx=vec3(g,"rx_center"); a=Float64(g["rx_radius"])
    scenario=="canonical" && return UnboundedDomain(rx,a)
    if scenario in ("concentric","eccentric")
        outer=vec3(g,"outer_center"); R=Float64(g["outer_radius"]); k=Float64(g["kappa"])
        length(outer)==3 && all(isfinite,outer) && k>=0 || error("Invalid outer geometry")
        norm(rx-outer)+a<R || error("Receiver not strictly inside outer sphere")
        concentric=norm(rx-outer)<1e-12
        (scenario=="concentric")==concentric || error("Scenario and centers disagree")
        # Dataset symmetry keys currently assume the eccentric axis is x.
        scenario=="eccentric" && (outer!=[0,0,0] || rx[2:3]!=[0,0]) && error("Eccentric dataset generator expects x-axis displacement")
        return FiniteShellRobinDomain(ShellGeometry(rx,outer,a,R,k,true),outer_absorb_probability(k,dt))
    end
    g["end_boundary"] in ("absorbing","reflecting") || error("Invalid end boundary")
    R=Float64(g["pipe_radius"]); lo=Float64(g["z_min"]); hi=Float64(g["z_max"]); k=Float64(g["wall_kappa"])
    R>0 && k>=0 && lo<hi || error("Invalid pipe parameters")
    radial_norm_xy(rx)+a<R && lo+a<rx[3]<hi-a || error("Receiver not inside pipe")
    rx[2]==0 && rx[3]==0 && lo==-hi || error("Dataset mirror keys require centered caps and RX y=z=0")
    return FinitePipeRobinDomain(PipeGeometry(R,lo,hi,k,g["end_boundary"],rx,a),outer_absorb_probability(k,dt))
end
