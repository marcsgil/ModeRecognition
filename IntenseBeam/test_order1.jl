using JLD2,Tullio

include("vizualization.jl")
include("initial_profiles.jl")

θ(string) = parse(Float32,string[findfirst('t',string)+1:findfirst('p',string)-1])
ϕ(string) = parse(Float32,string[findlast('i',string)+1:findfirst('.',string)-1])

function get_coef(path)
    Θ = θ(path) |> deg2rad
    Φ = ϕ(path) |> deg2rad
    [cos(Θ/2),sin(Θ/2)*cis(Φ)]
end
##
cd("/media/marcosgil/26601A1F6019F671/MLDatasets/StructuredLight/Experimental/d1")
paths = readdir()
rs = LinRange(-3f0,3f0,64)
basis = get_basis(1,rs)
##
path = rand(paths)
path
θ(path)
ϕ(path)
c = get_coef(path)

vizualize((@tullio ψ[i,j,k] := basis[i,j,k,l]*c[l]),ratio=4)
##
img = Float32.(load(path))
img/maximum(img) |> colorview(Gray)