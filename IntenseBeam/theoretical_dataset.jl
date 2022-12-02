using JLD2,Tullio
include("../initial_profiles.jl")
include("../vizualization.jl")

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

function generate_dataset(order,N,rs)
    c = generate_coeficients(order,N)
    basis = get_basis(order,rs)

    #In one line we perform the superposition!! (Einstein summation convention)
    @tullio x[i,j,m,n] := basis[i,j,m,k]*c[k,n] |> abs2

    x,format(c)
end
##
rs = LinRange(-3.5f0,3.5f0,64)
N = 1
vizualize(get_basis(N,rs),ratio=3)
x,y = generate_dataset(N,10^5,rs);
##
vizualize(x[:,:,:,1:3],ratio=8)
##
jldsave("C:/MLDatasets/StructuredLight/IntenseBeam/Order1/x_order_$N.jld2"; x)
jldsave("C:/MLDatasets/StructuredLight/IntenseBeam/Order1/y_order_$N.jld2"; y)