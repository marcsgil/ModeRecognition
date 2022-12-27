using Flux,JLD2

modified_sqrt(x) = √ifelse(x>0,x,zero(x))

f(x,y) = 1-x^2-y^2

function fidelity(ŷ,y)
    D = size(y,1)÷2
    r̂ = @view ŷ[1:D,:]
    î = @view ŷ[D+1:end,:]
    r = @view y[1:D,:]
    i = @view y[D+1:end,:]

    ĉ1 = modified_sqrt.(sum(map(f,r̂,î),dims=1))
    c1 = modified_sqrt.(sum(map(f,r,i),dims=1))

    R = sum((@. r̂*r+î*i), dims = 1) + ĉ1.*c1
    I = sum((@. r*î-r̂*i), dims = 1)

    sum( @. (R^2+I^2)/( (ĉ1^2+r̂^2+î^2)*(c1^2+r^2+i^2) ) )/size(y,2)
end

y = load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/y_32_photons.jld2")["y"]

fidelity(y,y)
@benchmark fidelity($y,$y)


function get_metrics(ŷ,y)
    #l = 0f0
    f = 0f0
    ntot = 0
    for (x, y) in loader
        x, y = x |> gpu, y |> gpu
        #l += loss(network(x|> gpu), y|> gpu) * size(x)[end]     
        f += fidelity(network(x), y) * size(x)[end]   
        ntot += size(x)[end]
    end
    (l/ntot),(f/ntot)
end