using LinearAlgebra, Statistics,JLD2,CairoMakie,Flux

function slow_fidelity(ŷ,y)
    D = size(y,1)÷2 + 1
    N = size(y,2)

    ĉs = Array{ComplexF32}(undef,D,N)  
    cs = Array{ComplexF32}(undef,D,N)

    for n in 1:N
        for m in 1:D-1
            cs[m+1,n] = y[m,n] + im*y[m+D-1,n]
            ĉs[m+1,n] = ŷ[m,n] + im*ŷ[m+D-1,n]
        end
        cs[1,n] = √max(0,(1-sum(abs2,@view cs[2:end,n])))
        ĉs[1,n] = √max(0,(1-sum(abs2,@view ĉs[2:end,n])))
    end
    [abs2(dot(view(cs,:,n),view(ĉs,:,n)))/(dot(view(cs,:,n),view(cs,:,n))*dot(view(ĉs,:,n),view(ĉs,:,n))) |> real for n in axes(cs,2)]
end

function slow_mae(ŷ,y)
    sum(abs,y.-ŷ,dims=1)./size(y,1) |> vec
end

best_model = load("WeakBeam/Results/Order1/$(N_photons)Photons/AdamW+MAE/best_model.jld2")["model"];

N_photons = 1024
x,y = load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/x_$(N_photons)_photons.jld2")["x"],
load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/y_$(N_photons)_photons.jld2")["y"]

ŷ = best_model(Float32.(x))
fs = slow_fidelity(ŷ,y)
mean(fs)
std(fs)

maes = slow_mae(ŷ,y)
mean(maes)
std(maes)

hist(fs,bins=[.98:.0001:1...])
hist(maes,bins=[0:.001:.1...])