function parse_options(args)
    d=Dict{String,Any}(); i=1
    while i<=length(args)
        key=args[i]
        startswith(key,"--") || error("Unexpected argument $key")
        key in ("--resume","--force") ? (d[key]=true;i+=1) : begin
            i<length(args) || error("Missing value for $key")
            d[key]=args[i+1]; i+=2
        end
    end
    return d
end
function main(args=ARGS)
    if isempty(args) || first(args) in ("--help","help")
        println("AMS-MOD: generate|validate|run|smoke --config CONFIG [--run-id ID --split all|train|validation|test --limit N --resume --max-new-repeats N --output-root PATH]")
        return
    end
    command=first(args); o=parse_options(args[2:end])
    allowed=Set(["--config","--run-id","--split","--limit","--resume","--max-new-repeats","--output-root","--profile","--force"])
    all(k->k in allowed,keys(o)) || error("Unknown option")
    haskey(o,"--config") || error("--config required")
    c=load_config(o["--config"])
    haskey(o,"--output-root") && (c["outputs"]["root"]=abspath(o["--output-root"]))
    if command=="generate"
        p=generate_dataset(c;force=get(o,"--force",false))
        println(json_value(Dict("scenario"=>c["scenario"],"positions"=>length(p),"manifest"=>c["dataset"]["manifest"])))
    elseif command=="validate"
        p=read_dataset(c["dataset"]["manifest"]); validate_dataset(c,p)
        println(json_value(Dict("scenario"=>c["scenario"],"valid"=>true,"positions"=>length(p),"counts"=>Dict(k=>count(p->p.split==k,p) for k in ("train","validation","test")),"integer_test"=>all(p->p.split!="test" || all(isinteger,p.x),p))))
    elseif command in ("run","smoke")
        command=="smoke" && get(o,"--profile","smoke")!="smoke" && error("smoke cannot select formal profile")
        run_dataset(c;profile=command=="smoke" ? "smoke" : get(o,"--profile","formal"),
                    run_id=get(o,"--run-id",Dates.format(now(),"yyyymmdd_HHMMSS")),split=get(o,"--split","all"),
                    limit=haskey(o,"--limit") ? parse(Int,o["--limit"]) : nothing,resume=get(o,"--resume",false),
                    max_new_repeats=haskey(o,"--max-new-repeats") ? parse(Int,o["--max-new-repeats"]) : typemax(Int))
    else
        error("Unknown command $command")
    end
end
