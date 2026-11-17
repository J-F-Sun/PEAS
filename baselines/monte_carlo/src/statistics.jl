"""Aggregate independent AMS repetitions, accounting for within-repeat cloning."""
function summarize_repeats(records,times)
    n=length(records); n>=2 || error("At least two independent repetitions required")
    all(r->r["converged"],records) || error("Cannot aggregate an unfinished AMS repetition")
    curves=zeros(length(times),n)
    for (j,r) in enumerate(records)
        absorption_curve!(view(curves,:,j),Float64(r["q_hat"]),Float64.(r["tau"]),times)
    end
    mu=vec(sum(curves;dims=2))./n
    sd=sqrt.(vec(sum((curves.-mu).^2;dims=2))./(n-1))
    se=sd./sqrt(n)
    return Dict("times"=>times,"q"=>mu,"q_std"=>sd,"q_se"=>se,
                "ci95_low"=>max.(0.0,mu.-1.96se),"ci95_high"=>min.(1.0,mu.+1.96se),
                "completed_repeats"=>n,"ci_method"=>"normal approximation across independent AMS repetitions")
end
