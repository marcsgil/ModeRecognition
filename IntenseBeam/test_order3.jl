using JLD2,Tullio

include("vizualization.jl")
include("initial_profiles.jl")

function get_radius_and_angle(string)
    positions = vcat(findall('r',string),findall('d',string),findall('.',string))

    L = length(positions)

    rs = Array{Float32}(undef,L÷2)
    ϕs = similar(rs)

    for n in eachindex(rs)
        rs[n] = parse(Float32,string[positions[n]+3:positions[n+1]-1])
    end

    ϕs[1] = 0

    for n in 2:L÷2
        ϕs[n] = parse(Float32,string[positions[n-1+L÷2]+3:positions[n+L÷2]-1])
    end

    rs,ϕs
end

polar2coef(r,ϕ) = r*cis(ϕ |> deg2rad)

function get_coef(path)
    map(polar2coef,get_radius_and_angle(path)...)
end
##
cd("/media/marcosgil/26601A1F6019F671/MLDatasets/StructuredLight/Experimental/d3")
paths = readdir()
rs = LinRange(-3.5f0,3.5f0,64)
basis = get_basis(3,rs)
##
path = rand(paths)
path
c = get_coef(path)

vizualize((@tullio ψ[i,j,k] := basis[i,j,k,l]*c[l]),ratio=4)
##
img = Float32.(load(path))
img/maximum(img) |> colorview(Gray)