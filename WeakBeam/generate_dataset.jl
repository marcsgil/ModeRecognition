using Distributions,JLD2,Tullio
include("../vizualization.jl")
include("../initial_profiles.jl")

function get_c1(c)
    sqrt.(ones(real(eltype(c)),1,size(c,2)).-sum(abs2,c,dims=1))
end

function generate_coeficients(order,N,type = ComplexF32)
    #Generates a random array that represents a series of modes superpsitions
    result = Array{type}(undef,order+1,N)

    #First we generate points on a order-sphere. 
    #We apply abs to be certain that all coordinates are posive.
    rs = abs.(randn(real(type),order,N))
    for n in axes(rs,2)
        rs[:,n] *= 1/√sum(abs2,rs[:,n])
    end

    #We multiply by a uniform random number to the power of 1/order to get a point on the interior of the order-sphere.
    #Then, we multiply by a random phase.
    result[2:end,:] = rand(real(type),order,N).^(1/order).*rs.*cispi.(2*rand(real(type),order,N))

    #=We choose the phase of the first coeficient as 0, which, together with the normalization condition,
    completely determines it.=# 
    result[1,:] = get_c1(@view result[2:end,:])
    result
end

function format(c)
    #Represents the coeficients c as a real array.
    #We stack the real and then the imaginary part.
    D = size(c,1)
    result = Array{real(eltype(c))}(undef,2D-2,size(c,2))
    result[1:D-1,:] = real.(@view c[2:end,:])
    result[D:2D-2,:] = imag.(@view c[2:end,:]) 
    result
end

function form_image(counts,imgsize)
    image = Array{Float32}(undef,imgsize,imgsize)
    N = length(counts)
    Threads.@threads for n in eachindex(image)
        image[n] = count(x->x==n,counts)/N
    end
    image
end

normalizeAndVec(x) = vec(x)/sum(x)

function generate_dataset(order,N_images,N_photons,rs)
    c = generate_coeficients(order,N_images)
    basis = get_basis(order,rs)

    @tullio x[i,j,m,n] := basis[i,j,m,k]*c[k,n] |> abs2
    support = 1:length(rs)^2

    Threads.@threads for n in axes(c,2)
        Dd = DiscreteNonParametric(support, normalizeAndVec(view(x,:,:,1,n)))
        Da = DiscreteNonParametric(support, normalizeAndVec(view(x,:,:,2,n)))

        x[:,:,1,n] = form_image(rand(Dd,N_photons),length(rs))
        x[:,:,2,n] = form_image(rand(Da,N_photons),length(rs))
    end

    x,format(c)
end
##
rs = LinRange(-4,4,64)
N_photons = 512
x,y = generate_dataset(1,10^5,N_photons,rs)
#vizualize(view(x,:,:,:,1:3),ratio=8)
jldsave("C:/MLDatasets/StructuredLight/WeakBeam/Order1/x_$(N_photons)_photons.jld2"; x)
jldsave("C:/MLDatasets/StructuredLight/WeakBeam/Order1/y_$(N_photons)_photons.jld2"; y)