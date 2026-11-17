function atomic_toml(path,data)
    mkpath(dirname(path))
    temp=path*".tmp"
    open(temp,"w") do io
        TOML.print(io,data;sorted=true)
    end
    Base.Filesystem.rename(temp,path)
end
function atomic_text(writer,path)
    mkpath(dirname(path)); temp=path*".tmp"
    open(writer,temp,"w")
    Base.Filesystem.rename(temp,path)
end
function json_value(x)
    x isa AbstractDict && return "{"*join([json_value(string(k))*":"*json_value(v) for (k,v) in sort(collect(x);by=first)],",")*"}"
    x isa AbstractVector && return "["*join(json_value.(x),",")*"]"
    x isa AbstractString && return "\""*replace(x,"\\"=>"\\\\","\""=>"\\\"","\n"=>"\\n","\r"=>"\\r","\t"=>"\\t")*"\""
    x isa Bool && return x ? "true" : "false"
    x === nothing && return "null"
    x isa AbstractFloat && !isfinite(x) && return "null"
    return string(x)
end
write_json(path,data)=atomic_text(io->print(io,json_value(data),'\n'),path)
function canonical_text(x)
    io=IOBuffer(); TOML.print(io,x;sorted=true); return String(take!(io))
end
struct DatasetPoint
    id::String
    split::String
    x::Vector{Float64}
end
function write_dataset(path,points,c)
    atomic_text(path) do io
        println(io,"id,split,x,y,z,distance_rx,gap_rx,gap_outer,band")
        for p in points
            d=norm(p.x-vec3(c["geometry"],"rx_center"))
            fields=[p.id,p.split,p.x...,d,d-c["geometry"]["rx_radius"],boundary_gap(c,p.x),distance_band(c,d)]
            println(io,join(fields,','))
        end
    end
end
function read_dataset(path)
    lines=readlines(path)
    !isempty(lines) && first(lines)=="id,split,x,y,z,distance_rx,gap_rx,gap_outer,band" || error("Unexpected dataset header")
    points=DatasetPoint[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        v=split(line,','); length(v)==9 || error("Malformed dataset row")
        push!(points,DatasetPoint(v[1],v[2],parse.(Float64,v[3:5])))
    end
    return points
end
