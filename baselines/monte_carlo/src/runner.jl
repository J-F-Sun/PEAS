function run_signature(c,s,profile,points)
    payload=Dict("framework_version"=>VERSION,"julia_version"=>string(Base.VERSION),"seed_scheme"=>"sha256_numeric_v1",
                 "scenario"=>c["scenario"],"profile"=>profile,"geometry"=>c["geometry"],"simulation"=>s,
                 "points"=>[Dict("id"=>p.id,"split"=>p.split,"x"=>p.x) for p in points])
    return bytes2hex(sha256(canonical_text(payload)))
end
function repeat_record(domain,p,s,c,k)
    seed=stable_seed(s["master_seed"],c["scenario"],p.id,p.x...,k)
    started=time()
    result=run_single(domain=domain,x0=p.x,dt=Float64(s["dt"]),t_max=Float64(s["t_max"]),
                      num_replicas=Int(s["num_replicas"]),n_worst=Int(s["n_worst"]),seed=seed,
                      max_iterations=Int(s["max_iterations"]),times=[0.0,Float64(s["t_max"])])
    return Dict("repeat"=>k,"seed"=>seed,"q_hat"=>result.q_hat,"tau"=>result.tau,
                "iterations"=>result.iterations,"num_replica_total"=>result.num_replica_total,
                "num_final_success"=>result.num_final_success,"level_current"=>result.level_current,
                "converged"=>result.converged,"elapsed_seconds"=>time()-started)
end
function point_output(dir,p,c,s,profile,records)
    result=summarize_repeats(records,output_times(s))
    merge!(result,Dict("schema_version"=>1,"framework_version"=>VERSION,"scenario"=>c["scenario"],
                      "profile"=>profile,"data_role"=>"ams_mod_estimate","id"=>p.id,"split"=>p.split,
                      "x0"=>p.x,"geometry"=>c["geometry"],"simulation"=>s,
                      "total_replica_paths"=>sum(r["num_replica_total"] for r in records),
                      "elapsed_repeat_seconds"=>sum(r["elapsed_seconds"] for r in records)))
    write_json(joinpath(dir,"curve.json"),result)
    atomic_text(joinpath(dir,"curve.csv")) do io
        println(io,"time,q,q_std,q_se,ci95_low,ci95_high")
        for i in eachindex(result["times"])
            println(io,join([result[k][i] for k in ("times","q","q_std","q_se","ci95_low","ci95_high")],','))
        end
    end
    atomic_text(joinpath(dir,"repeat_summary.csv")) do io
        println(io,"repeat,seed,q_hat,iterations,num_replica_total,num_final_success,converged,elapsed_seconds")
        for r in records
            println(io,join([r[k] for k in ("repeat","seed","q_hat","iterations","num_replica_total","num_final_success","converged","elapsed_seconds")],','))
        end
    end
end
"""Process bounded batches of repetitions; persist each completed batch atomically."""
function run_dataset(c;profile="formal",run_id=Dates.format(now(),"yyyymmdd_HHMMSS"),
                     split="all",limit=nothing,resume=false,max_new_repeats=typemax(Int))
    occursin(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",run_id) || error("Unsafe run_id")
    split in ("all","train","validation","test") || error("Unknown split")
    max_new_repeats>=0 || error("Negative repetition limit")
    s=settings(c,profile); points=read_dataset(c["dataset"]["manifest"]); validate_dataset(c,points)
    selected=[p for p in points if split=="all" || p.split==split]
    if limit!==nothing
        limit>0 || error("limit must be positive")
        selected=selected[1:min(limit,length(selected))]
    elseif profile=="smoke"
        selected=selected[1:min(2,length(selected))]
    end
    dir=joinpath(c["outputs"]["root"],c["scenario"],run_id)
    meta_path=joinpath(dir,"run.toml"); signature=run_signature(c,s,profile,points)
    if isdir(dir)
        resume || error("Run exists; use --resume with the same config/profile")
        isfile(meta_path) || error("Missing run metadata")
        TOML.parsefile(meta_path)["signature"]==signature || error("Config, profile or dataset differs from saved run")
    else
        resume && error("Cannot resume a missing run")
        mkpath(dir)
        atomic_toml(meta_path,Dict("schema_version"=>1,"signature"=>signature,"framework_version"=>VERSION,
                                  "scenario"=>c["scenario"],"profile"=>profile,"geometry"=>c["geometry"],"simulation"=>s,
                                  "julia_version"=>string(Base.VERSION),"threads_at_creation"=>nthreads(),"machine"=>Sys.MACHINE,
                                  "created_at_local"=>string(now()),"dataset_counts"=>Dict(k=>c["dataset"][k] for k in ("train","validation","test"))))
        cp(c["config_path"],joinpath(dir,"config.toml"))
        cp(c["dataset"]["manifest"],joinpath(dir,"positions.csv"))
    end
    lock=joinpath(dir,".running")
    ispath(lock) && error("Run has a .running lock; check for an active process before clearing a stale lock")
    mkdir(lock)
    write(joinpath(lock,"pid.txt"),string(getpid()))
    domain=build_domain(c,Float64(s["dt"])); new_count=0; finished=0
    try
        for p in selected
            point_dir=joinpath(dir,p.split,p.id); path=joinpath(point_dir,"state.toml")
            records=Dict{String,Any}[]
            if isfile(path)
                saved=TOML.parsefile(path)
                saved["signature"]==signature && saved["id"]==p.id || error("Mismatched point checkpoint")
                records=Dict{String,Any}[x for x in saved["repeats"]]
                [x["repeat"] for x in records]==collect(1:length(records)) || error("Broken repeat sequence")
                all(x->x["converged"],records) || error("Saved repetition failed to converge: $(p.id). Inspect state.toml; use a new run with a revised budget.")
                length(records)<=s["repeats"] || error("Too many saved repeats")
            end
            while length(records)<s["repeats"] && new_count<max_new_repeats
                batchsize=min(c["outputs"]["checkpoint_batch"],nthreads(),s["repeats"]-length(records),max_new_repeats-new_count)
                batchsize>=1 || error("Invalid checkpoint batch")
                next=length(records)+1
                tasks=[Threads.@spawn repeat_record(domain,p,s,c,k) for k in next:next+batchsize-1]
                batch=fetch.(tasks) # fetch order fixes aggregation order across thread counts
                append!(records,batch); new_count+=batchsize
                atomic_toml(path,Dict("signature"=>signature,"id"=>p.id,"profile"=>profile,"x0"=>p.x,"repeats"=>records))
                @printf("[%s/%s] %s repeats=%d/%d new=%d\n",c["scenario"],profile,p.id,length(records),s["repeats"],new_count)
                flush(stdout)
                all(x->x["converged"],batch) || error("AMS max_iterations reached for $(p.id); saved diagnostic repetition, no final curve exported")
            end
            if length(records)==s["repeats"]
                point_output(point_dir,p,c,s,profile,records); finished+=1
            end
            new_count>=max_new_repeats && break
        end
        # Report coverage of the full fixed dataset, including previous selections.
        completed=count(p->isfile(joinpath(dir,p.split,p.id,"curve.json")),points)
        status=Dict("scenario"=>c["scenario"],"profile"=>profile,"run_id"=>run_id,"new_repeats"=>new_count,
                    "selected_positions"=>length(selected),"selected_completed"=>finished,
                    "completed_positions"=>completed,"dataset_positions"=>length(points),
                    "complete"=>completed==length(points),"path"=>dir)
        atomic_toml(joinpath(dir,"status.toml"),status)
        println(json_value(status))
        return status
    finally
        rm(lock;recursive=true)
    end
end
