using Distributions,Combinatorics,LinearAlgebra,ThreadsX

function get_samples(D,N,T=Float32)
    angles = Array{T}(undef,2(D-1),N)
    angles[1:D-1,:] = rand(Uniform(0,π),D-1,N)
    angles[D:end,:] = rand(Uniform(0,2π),D-1,N)
    angles
end

function stereographic_projection(θ,ϕ)
    cot(θ/2)*cis(ϕ)
end

function angles2roots(angles)
    D = size(angles,1)÷2+1
    θs = @view angles[1:D-1,:] 
    ϕs = @view angles[D:end,:]

    ThreadsX.map(stereographic_projection,θs,ϕs)
end

function roots2coefs(roots)
    D = size(roots,1)+1
    coefs = Array{complex(eltype(roots))}(undef, D, size(roots,2))

    Threads.@threads for k in axes(coefs,2)
        for j in axes(coefs,1)
           coefs[j,k] = sum(prod,combinations(view(roots,:,k),D-j))
        end
    end

    coefs
end

function fidelity_from_coefs(ĉ,c)
    ThreadsX.map((ĉ,c)->abs2(ĉ⋅c)/((ĉ⋅ĉ)*(c⋅c)),eachcol(ĉ),eachcol(c))
end

function fidelity(ângles,angles)
    fidelity_from_coefs(ângles |> angles2roots |> roots2coefs,angles |> angles2roots |> roots2coefs)
end
##

angles = get_samples(2,10^5)
angles2 = get_samples(2,10^5)
roots = angles2roots(angles)
ĉ = roots2coefs(roots)
fidelity_from_coefs(ĉ,ĉ)
fidelity(angles,angles)

@benchmark angles2roots($angles)
@benchmark roots2coefs($roots)
@benchmark fidelity_from_coefs($ĉ,$ĉ)

@benchmark fidelity(angles,angles)
sum(fidelity(angles,angles2))/size(angles,2)