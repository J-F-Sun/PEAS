# Counts are transmitter curves, not individual time samples or AMS replicas.
function boundary_gap(c,x)
    g=c["geometry"]
    c["scenario"]=="canonical" && return Inf
    c["scenario"]=="pipe" && return min(g["pipe_radius"]-hypot(x[1],x[2]),x[3]-g["z_min"],g["z_max"]-x[3])
    return g["outer_radius"]-norm(x-vec3(g,"outer_center"))
end
function valid_source(c,x)
    d=norm(x-vec3(c["geometry"],"rx_center")); s=c["dataset"]
    return s["min_distance"]-1e-10<=d<=s["max_distance"]+1e-10 && boundary_gap(c,x)>=s["boundary_margin"]-1e-10
end
function orbit_key(c,x)
    scenario=c["scenario"]; g=c["geometry"]
    v= scenario in ("canonical","concentric") ? [sum(abs2,x-vec3(g,"rx_center"))] :
       scenario=="eccentric" ? [x[1],x[2]^2+x[3]^2] : [x[1],abs(x[2]),abs(x[3])]
    return Tuple(round.(v;digits=10))
end
function space_descriptor(c,x)
    s=c["dataset"]; scenario=c["scenario"]
    scenario in ("canonical","concentric") && return [norm(x-vec3(c["geometry"],"rx_center"))/s["max_distance"]]
    scenario=="eccentric" && return [x[1]/c["geometry"]["outer_radius"],hypot(x[2],x[3])/c["geometry"]["outer_radius"]]
    return [x[1]/c["geometry"]["pipe_radius"],abs(x[2])/c["geometry"]["pipe_radius"],abs(x[3])/s["max_distance"]]
end
distance_band(c,d)= d<c["dataset"]["near_end"] ? "near" : d<c["dataset"]["middle_end"] ? "middle" : "far"
function choose_spread(c,pool,n,used;anchors=Vector{Vector{Float64}}())
    available=[x for x in pool if !(orbit_key(c,x) in used)]
    # Greedy maximin in symmetry-reduced geometry, retaining each band quota.
    descriptors=space_descriptor.(Ref(c),available)
    scores=fill(Inf,length(available))
    for a in anchors
        f=space_descriptor(c,a)
        for i in eachindex(available); scores[i]=min(scores[i],sum(abs2,descriptors[i]-f)); end
    end
    chosen=Vector{Vector{Float64}}()
    for j in 1:n
        isempty(available) && error("Not enough distinct source classes")
        index=argmax(scores); x=available[index]; push!(chosen,x); push!(used,orbit_key(c,x))
        f=descriptors[index]
        deleteat!(available,index); deleteat!(descriptors,index); deleteat!(scores,index)
        for i in eachindex(available); scores[i]=min(scores[i],sum(abs2,descriptors[i]-f)); end
    end
    return chosen
end
function integer_pool(c)
    s=c["scenario"]; pool=Vector{Vector{Float64}}(); used=Set{Tuple}()
    if s=="canonical"
        return [[Float64(d),0.0,0.0] for d in 2:21]
    elseif s=="concentric"
        for x in 0:9,y in 0:x,z in 0:y
            p=Float64[x,y,z]
            if valid_source(c,p) && !(orbit_key(c,p) in used)
                push!(pool,p); push!(used,orbit_key(c,p))
            end
        end
    elseif s=="eccentric"
        limit=floor(Int,c["geometry"]["outer_radius"]-c["dataset"]["boundary_margin"])
        for x in -limit:limit,y in 0:limit
            p=Float64[x,y,0]
            valid_source(c,p) && push!(pool,p)
        end
    else
        for x in -4:4,y in 0:4,z in 0:9
            p=Float64[x,y,z]
            valid_source(c,p) && push!(pool,p)
        end
    end
    return pool
end
function continuous_pools(c,rng)
    ds=c["dataset"]; pools=Dict(b=>Vector{Vector{Float64}}() for b in ("near","middle","far"))
    edges=(ds["min_distance"],ds["near_end"],ds["middle_end"],ds["max_distance"])
    rx=vec3(c["geometry"],"rx_center"); count=ds["candidates_per_band"]
    for (j,band) in enumerate(("near","middle","far"))
        lo,hi=edges[j],edges[j+1]
        attempts=0; seen=Set{Tuple}()
        while length(pools[band])<count
            attempts+=1; attempts<1000000 || error("Cannot fill candidate pool")
            if c["scenario"]=="concentric"
                r=lo+(hi-lo)*rand(rng); p=Float64[r,0,0]
            else
                r=cbrt(lo^3+rand(rng)*(hi^3-lo^3))
                u=2rand(rng)-1; phi=2pi*rand(rng); v=sqrt(1-u^2)
                p=rx+r.*[v*cos(phi),v*sin(phi),u]
            end
            p=round.(p;digits=6)
            key=orbit_key(c,p)
            if valid_source(c,p) && distance_band(c,norm(p-rx))==band && !(key in seen)
                push!(pools[band],p); push!(seen,key)
            end
        end
    end
    return pools
end
function quotas(n)
    near=round(Int,.45n); mid=round(Int,.35n)
    return Dict("near"=>near,"middle"=>mid,"far"=>n-near-mid)
end
function generate_dataset(c;force=false)
    path=c["dataset"]["manifest"]
    if isfile(path) && !force
        p=read_dataset(path); validate_dataset(c,p); return p
    end
    rng=Xoshiro(c["dataset"]["seed"]); used=Set{Tuple}(); result=DatasetPoint[]
    all_chosen=Vector{Vector{Float64}}()
    sets=Dict{String,Vector{Vector{Float64}}}()
    test=integer_pool(c)
    if c["scenario"]=="canonical"
        length(test)==c["dataset"]["test"] || error("Canonical integer test rule expects 20 points")
        sets["test"]=test
        foreach(p->push!(used,orbit_key(c,p)),test)
        # Uniform radial design on the enlarged historical interval.
        ds=c["dataset"]
        pool=[[round(ds["min_distance"]+(ds["max_distance"]-ds["min_distance"])*(i+.37)/2001;digits=6),0.0,0.0] for i in 0:2000]
        sets["validation"]=choose_spread(c,pool,ds["validation"],used)
        sets["train"]=choose_spread(c,pool,ds["train"],used;anchors=sets["validation"])
    else
        reserved=c["scenario"]=="concentric" ? [Float64[d,0,0] for d in 2:9] :
                 c["scenario"]=="eccentric" ? [Float64[-1,0,0],Float64[0,0,0],Float64[1,0,0]] :
                 [Float64[1,0,-4],Float64[0,0,-3],Float64[-1,0,-2]]
        all(p->valid_source(c,p),reserved) || error("Reserved comparison point is outside dataset region")
        foreach(p->push!(used,orbit_key(c,p)),reserved)
        chosen=copy(reserved)
        # 7 near / 7 middle / 6 far for the 20-point evaluation set.
        qtest=Dict("near"=>7,"middle"=>7,"far"=>6)
        for band in ("near","middle","far")
            subset=[p for p in test if distance_band(c,norm(p-vec3(c["geometry"],"rx_center")))==band]
            n=qtest[band]-count(p->distance_band(c,norm(p-vec3(c["geometry"],"rx_center")))==band,reserved)
            append!(chosen,choose_spread(c,subset,n,used;anchors=chosen))
        end
        length(chosen)==c["dataset"]["test"] || error("Extended integer test rule expects 20 points")
        sets["test"]=chosen; append!(all_chosen,chosen)
        pools=continuous_pools(c,rng)
        for split in ("validation","train")
            pts=Vector{Vector{Float64}}()
            for band in ("near","middle","far")
                selected=choose_spread(c,pools[band],quotas(c["dataset"][split])[band],used;anchors=vcat(all_chosen,pts))
                append!(pts,selected)
            end
            sets[split]=pts; append!(all_chosen,pts)
        end
    end
    for split in ("train","validation","test")
        sort!(sets[split];by=x->(norm(x-vec3(c["geometry"],"rx_center")),Tuple(x)))
        for (i,x) in enumerate(sets[split]); push!(result,DatasetPoint(@sprintf("%s_%s_%03d",c["scenario"],split,i),split,x)); end
    end
    validate_dataset(c,result); write_dataset(path,result,c)
    atomic_toml(replace(path,".csv"=>"_design.toml"),Dict("schema_version"=>1,"dataset_seed"=>c["dataset"]["seed"],"scenario"=>c["scenario"],"dataset"=>TOML.parsefile(c["config_path"])["dataset"],"geometry"=>c["geometry"],"selection"=>"held-out integer tests; radial/geometry maximin train-validation; symmetry classes disjoint"))
    return result
end
function validate_dataset(c,points)
    length(unique(p.id for p in points))==length(points) || error("Duplicate IDs")
    all(p->p.split in ("train","validation","test"),points) || error("Unknown split")
    for split in ("train","validation","test")
        count(p->p.split==split,points)==c["dataset"][split] || error("Wrong $split count")
    end
    domain=build_domain(c,c["simulation"]["dt"])
    for p in points
        validate_domain(domain,p.x); valid_source(c,p.x) || error("Point outside source distribution: $(p.id)")
        p.split=="test" && !all(isinteger,p.x) && error("Test point must have integer coordinates")
    end
    length(unique(orbit_key(c,p.x) for p in points))==length(points) || error("Repeated physical symmetry class within or across splits")
    return true
end
