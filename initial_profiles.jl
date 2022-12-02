using ClassicalOrthogonalPolynomials:laguerrel,hermiteh
using Parameters:@unpack

function set_max_intensity!(ψ,I_max)
    current_I_max = maximum(abs2,ψ)
    ψ *= √(I_max/current_I_max)
end

function lens(x,y,fx,fy;k=1)
	exp( -im*k*( x^2/fx + y^2/fy )/2 )
end

#Laguerre-Gaussian

Base.@kwdef struct LGConfig{T1 <: Integer, T2 <: Integer, T3 <: Number}
    p::T1 = 0
    l::T2 = 0
    γ₀::T3 = 1
    k::T3 = 1
end

order(config::LGConfig) = 2config.p + abs(config.l)

normalization(config::LGConfig) = 1/(config.γ₀*√( π*prod(config.p+1:config.p+abs(config.l))))

function core_LG(x,y,α,γ₀,p,l)
    r2 = (x^2 + y^2)/γ₀^2
    α*exp(-α*r2/2)*(abs(α)*(x+im*sign(l)*y)/γ₀)^abs(l)*laguerrel(p,abs(l),abs2(α)*r2)
end

function LG(xs::AbstractArray,ys::AbstractArray,z::Number=0,config=LGConfig())
    @unpack p,l,γ₀,k = config
    α = convert(complex(eltype(xs)),1/(1+im*z/(k*γ₀^2)))
    prefactor = cis(order(config)*angle(α))
    
    map(r->prefactor*core_LG(r...,α,γ₀,p,l),Iterators.product(xs,ys))
end

function LG(xs::AbstractArray,ys::AbstractArray,zs::AbstractArray,config=LGConfig())
    @unpack p,l,γ₀,k = config
    result = Array{complex(eltype(xs))}(undef, length(xs),length(ys),length(zs))
    transverse_grid = Iterators.product(xs,ys) |> collect

    Threads.@threads for n in axes(result,3)
        α = convert(complex(eltype(xs)),1/(1+im*zs[n]/(k*γ₀^2)))
        prefactor = cis(order(config)*angle(α))
    
        map!(r->prefactor*core_LG(r...,α,γ₀,p,l),view(result,:,:,n),transverse_grid)
    end

    result
end

#Hermite Gaussian

Base.@kwdef struct HGConfig{T1 <: Integer, T2 <: Integer, T3 <: Number}
    m::T1 = 0
    n::T2 = 0
    γ₀::T3 = 1
    k::T3 = 1
end

order(config::HGConfig) = config.m + config.n

normalization(config::HGConfig) = 1/(config.γ₀*√( π*2^(order(config))*factorial(config.n)*factorial(config.m)))

function core_HG(x,y,α,γ₀,m,n)
    ξ = x/γ₀
    η = y/γ₀
    α*exp(-α*(ξ^2+η^2)/2)*hermiteh(m,abs(α)*ξ)*hermiteh(n,abs(α)*η)
end

function HG(xs::AbstractArray,ys::AbstractArray,z::Number=0,config=HGConfig())
    @unpack m,n,γ₀,k = config
    α = convert(complex(eltype(xs)),1/(1+im*z/(k*γ₀^2)))
    prefactor = cis(order(config)*angle(α))
    
    map(r->prefactor*core_HG(r...,α,γ₀,m,n),Iterators.product(xs,ys))
end

function HG(xs::AbstractArray,ys::AbstractArray,zs::AbstractArray,config=HGConfig())
    @unpack m,n,γ₀,k = config
    result = Array{complex(eltype(xs))}(undef, length(xs),length(ys),length(zs))
    transverse_grid = Iterators.product(xs,ys) |> collect

    Threads.@threads for i in axes(result,3)
        α = convert(complex(eltype(xs)),1/(1+im*zs[i]/(k*γ₀^2)))
        prefactor = cis(order(config)*angle(α))
    
        map!(r->prefactor*core_HG(r...,α,γ₀,m,n),view(result,:,:,i),transverse_grid)
    end

    result
end
##
function core_diag_HG(x,y,α,γ₀,m,n)
    ξ = (x+y)/(√2γ₀)
    η = (x-y)/(√2γ₀)
    α*exp(-α*(ξ^2+η^2)/2)*hermiteh(m,abs(α)*ξ)*hermiteh(n,abs(α)*η)
end

function diag_HG(xs::AbstractArray,ys::AbstractArray,z::Number=0,config=HGConfig())
    @unpack m,n,γ₀,k = config
    α = convert(complex(eltype(xs)),1/(1+im*z/(k*γ₀^2)))
    prefactor = cis(order(config)*angle(α))
    
    map(r->prefactor*core_diag_HG(r...,α,γ₀,m,n),Iterators.product(xs,ys))
end

function diag_HG(xs::AbstractArray,ys::AbstractArray,zs::AbstractArray,config=HGConfig())
    @unpack m,n,γ₀,k = config
    result = Array{complex(eltype(xs))}(undef, length(xs),length(ys),length(zs))
    transverse_grid = Iterators.product(xs,ys) |> collect

    Threads.@threads for i in axes(result,3)
        α = convert(complex(eltype(xs)),1/(1+im*zs[i]/(k*γ₀^2)))
        prefactor = cis(order(config)*angle(α))
    
        map!(r->prefactor*core_diag_HG(r...,α,γ₀,m,n),view(result,:,:,i),transverse_grid)
    end

    result
end

function get_basis(order,rs)
    #Construct the basis modes of given order.
    #The third dimension runs over the direct basis (LG) and the astigmatic basis (diagonal HG)
    #The modes are calculated over a grid defined by rs

    basis = Array{complex(eltype(rs))}(undef,length(rs),length(rs),2,order+1)

    for k in 0:order
        #config_lg = LGConfig(p = min(k,order-k),l = order-2k)
        #config_hg = HGConfig(m=order-k,n = k)
        config_lg = LGConfig(p = min(k,order-k),l = 2k-order)
        config_hg = HGConfig(m=k,n = order-k)
        basis[:,:,1,k+1] = LG(rs,rs,0,config_lg)
        basis[:,:,1,k+1] *= 1/√sum(abs2,basis[:,:,1,k+1])
        basis[:,:,2,k+1] = diag_HG(rs,rs,0,config_hg)
        basis[:,:,2,k+1] *= 1/√sum(abs2,basis[:,:,2,k+1])
    end    

    basis
end