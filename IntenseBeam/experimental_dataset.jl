using JLD2,Tullio,LinearAlgebra

include("../vizualization.jl")
include("../initial_profiles.jl")

function center_of_mass(M)
    i_cm = zero(eltype(M))
    j_cm = zero(eltype(M))
    for j in axes(M,2)
        for i in axes(M,1)
            i_cm += i*M[i,j]
            j_cm += j*M[i,j]
        end
    end
    mass = sum(M)
    round(Int,(i_cm/mass).val),round(Int,(j_cm/mass).val)
end

function centralize(M,d)
    i,j = center_of_mass(M)
    view(M,i-d÷2+1:i+d÷2,j-d÷2+1:j+d÷2)
end

function format(paths,d,L)
    x = Array{Float32}(undef,L,L,2,length(paths))
    Threads.@threads for n in eachindex(paths)
        X = reshape(load(paths[n]),1200,800,2)
        x[:,:,1,n] = imresize(centralize(view(X,:,:,1),d),(L,L))
        x[:,:,2,n] = imresize(centralize(view(X,:,:,2),d),(L,L))
    end
    x
end
##
#Order 1

paths = readdir("D:/MLDatasets/StructuredLight/Experimental/d3")
x = format(paths,600,64)
N = rand(eachindex(paths))
x[:,:,1,N] |> vizualize
##
jldsave("D:/MLDatasets/StructuredLight/Experimental/x_order2.jld2"; x)
##

θ(string) = parse(Float32,string[findfirst('t',string)+1:findfirst('p',string)-1])
ϕ(string) = parse(Float32,string[findlast('i',string)+1:findfirst('.',string)-1])
##
test = paths[4321]
θ(test)
ϕ(test)
##
function get_labels1(names)
    y = Array{Float32}(undef,2,length(names))
    for n in eachindex(names)
        st = sind(θ(names[n])/2)
        sf,cf = sincosd(ϕ(names[n]))
        
        y[1,n] = st*cf
        y[2,n] = st*sf
    end
    y
end
y = get_labels(paths)
paths[27]
y[:,27]
##
jldsave("D:/MLDatasets/StructuredLight/Experimental/y_order2.jld2"; y)
##
#Order 2
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

function get_formated_coef(path)
    rs,ϕs = get_radius_and_angle(path)
    normalize!(rs)
    cs = map(polar2coef,(@view rs[2:end]),(@view ϕs[2:end]))
    vcat(real.(cs),imag.(cs))
end

function get_labels3(paths)
    y = Array{Float32}(undef,6,length(paths))
    Threads.@threads for n in eachindex(paths)
        y[:,n] = get_formated_coef(paths[n])
    end
    y
end
##
cd("D:/MLDatasets/StructuredLight/Experimental/d3/")
paths = readdir()

y = get_labels3(paths)
jldsave("D:/MLDatasets/StructuredLight/Experimental/y_order3.jld2"; y)
##
x = format(paths,600,64)
@btime format(paths[1:100],600,64)
N = rand(axes(x,4))
x |> vizualize
jldsave("D:/MLDatasets/StructuredLight/Experimental/x_order3.jld2"; x)
##
load("D:/MLDatasets/StructuredLight/Experimental/x_order3.jld2")["x"]
##
#Order 4
function get_labels4(paths)
    y = Array{Float32}(undef,8,length(paths))
    Threads.@threads for n in eachindex(paths)
        y[:,n] = get_formated_coef(paths[n])
    end
    y
end

cd("D:/MLDatasets/StructuredLight/Experimental/d4/")
paths = readdir()

y = get_labels4(paths)
jldsave("D:/MLDatasets/StructuredLight/Experimental/y_order4.jld2"; y)
##
x = format(paths,700,64)
@btime format(paths[1:100],600,64)
N = rand(axes(x,4))
vizualize(x[:,:,:,N:N+4],ratio=4)
jldsave("D:/MLDatasets/StructuredLight/Experimental/x_order4.jld2"; x)