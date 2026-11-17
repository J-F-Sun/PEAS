# Stable numeric streams, independent of scheduling and point enumeration order.
# Pin the Julia version for bit-for-bit reruns of Random.Xoshiro streams.
function stable_seed(parts...)
    io=IOBuffer()
    for part in parts
        value=string(part)
        print(io,ncodeunits(value),':',value,';')
    end
    digest=sha256(take!(io))
    value=UInt64(0)
    for i in 1:8
        value=(value<<8)|UInt64(digest[i])
    end
    return Int(value & UInt64(typemax(Int)))
end
seed_for(policy::Symbol,parts...) = policy==:legacy ? hash(tuple(parts...)) : stable_seed(parts...)
