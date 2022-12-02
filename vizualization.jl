using Images

function normalize(ψ)
    ψ/maximum(ψ)
end

function vizualize(ψ::AbstractArray{T,2};ratio = 1) where T <: Real
    imresize(normalize(ψ)',ratio=ratio) |> colorview(Gray)
end

function vizualize(ψ::AbstractArray{T,3};ratio = 1) where T <: Real
    hcat((vizualize(slice,ratio=ratio) for slice in eachslice(ψ,dims=3))...)
end

function vizualize(ψ::AbstractArray{T,4};ratio = 1) where T <: Real
    vcat((vizualize(slice,ratio=ratio) for slice in eachslice(ψ,dims=3))...)
end

function vizualize(ψ;ratio = 1)
    vizualize(abs2.(ψ),ratio=ratio)
end