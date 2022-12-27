using Distributions,Combinatorics,LinearAlgebra
using Symbolics
using Flux
using Statistics
using CUDA
CUDA.allowscalar(false)

function get_samples(D,N,T=Float32)
    angles = Array{T}(undef,2(D-1),N)
    angles[1:D-1,:] = rand(Uniform(0,π),D-1,N)
    angles[D:end,:] = rand(Uniform(0,2π),D-1,N)
    angles
end
##
function build_fidelity()
    @variables θ ϕ θ̂ ϕ̂
    stereographic_projection(θ,ϕ) = cot(θ/2)*cis(ϕ)

    c = [sum(prod,combinations(stereographic_projection(θ,ϕ),length(θ)-n)) for n in 0:length(θ)]
    ĉ = [sum(prod,combinations(stereographic_projection(θ̂,ϕ̂),length(θ)-n)) for n in 0:length(θ)]

    f = build_function( abs2(c⋅ĉ)/(sum(abs2,ĉ)*sum(abs2,c)) ,θ̂,ϕ̂,θ,ϕ) |> eval

    function fidelity(ŷ,y)
        s = size(y,1)
        θ̂ = @view ŷ[1:s÷2,:]
        ϕ̂ = @view ŷ[s÷2+1:end,:]
        θ = @view y[1:s÷2,:]
        ϕ = @view y[s÷2+1:end,:]
    
        map(f,θ̂,ϕ̂,θ,ϕ)
    end
end
##
fidelity = build_fidelity()

angles = get_samples(2,10^5)
angles2 = get_samples(2,10^5)

gpu_angles = angles |> gpu
gpu_angles2 = angles2 |> gpu

fidelity(angles,angles2)
fidelity(gpu_angles,gpu_angles2)

@benchmark fidelity($angles,$angles2)
@benchmark fidelity($gpu_angles,$gpu_angles2)
##
function build_fidelity(D)
    Meta.parse("@variables"*prod(n->" θ$n ϕ$n θ̂$n ϕ̂$n",1:D-1)) |> eval

    θ =  ntuple(n->Meta.parse("θ$n")|> eval, D-1)
    ϕ =  ntuple(n->Meta.parse("ϕ$n")|> eval, D-1)
    θ̂ =  ntuple(n->Meta.parse("θ̂$n")|> eval, D-1)
    ϕ̂ =  ntuple(n->Meta.parse("ϕ̂$n")|> eval, D-1)

    stereographic_projection(θ,ϕ) = @. cot(θ/2)*cis(ϕ)

    c = [sum(prod,combinations(stereographic_projection(θ,ϕ),length(θ)-n)) for n in 0:length(θ)]
    ĉ = [sum(prod,combinations(stereographic_projection(θ̂,ϕ̂),length(θ)-n)) for n in 0:length(θ)]

    f = build_function( abs2(c⋅ĉ)/(sum(abs2,ĉ)*sum(abs2,c)),θ̂ ...,ϕ̂ ...,θ...,ϕ...) |> eval

    function fidelity(ŷ,y)
        map(f,ntuple(n->view(ŷ,n,:),size(ŷ,1))...,ntuple(n->view(y,n,:),size(y,1))...)
    end
end
##
D = 5
N= 10^5
fidelity = build_fidelity(D)

angles = get_samples(D,N) |> gpu
angles2 = get_samples(D,N) |> gpu
##
fidelity(angles,angles2)
@benchmark fidelity($angles,$angles2)