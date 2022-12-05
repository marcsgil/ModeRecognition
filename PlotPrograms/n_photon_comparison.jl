using CairoMakie,JLD2,Flux,LinearAlgebra,Statistics

function fidelity(ŷ,y)
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

function my_error(ŷ,y)
    sum(abs,y.-ŷ,dims=1)./size(y,1) |> vec
end
##

Ns_photons = [2^n for n in 5:10]
means_f = Vector{Float32}(undef,6)
stds_f = Vector{Float32}(undef,6)
means_e = Vector{Float32}(undef,6)
stds_e = Vector{Float32}(undef,6)

best_model = load("IntenseBeam/Results/Order1/model.jld2")["model"]

for (n,N_photons) in enumerate(Ns_photons)

    #best_model = load("WeakBeam/Results/Order1/$(N_photons)Photons/AdamW+MAE/best_model.jld2")["model"];
    x = load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/xtest_$(N_photons)_photons.jld2")["xtest"]
    y = load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/ytest_$(N_photons)_photons.jld2")["ytest"]

    ŷ = best_model(x)
    fs = fidelity(ŷ,y)
    means_f[n] =  round(mean(fs),sigdigits=3)
    stds_f[n] = round(std(fs),sigdigits=2)

    errors = my_error(ŷ,y)
    means_e[n] = round(mean(errors),sigdigits=2)
    stds_e[n] = round(std(errors),sigdigits=2)
end
##
fig = Figure(resolution=(1600,800),fontsize=28)
ax1 = Axis(fig[1,1],xlabel = L"\log_2 (\text{number of photons})", ylabel = "Mean Fidelity",yticks=.76:.02:1,xticks=5:10)
ax2 = Axis(fig[1,2],xlabel = L"\log_2 (\text{number of photons})", ylabel = "MAE",yticks=0:.04:.30,xticks=5:10)
#ylims!(ax,0.968,1)
#xlims!(5,10)
lines!(ax1,5:10,means_f,color=:red,linewidth=4)
errorbars!(ax1,5:10,means_f, stds_f,ones(6)-means_f, color = :black, whiskerwidth = 10,linewidth=3)

lines!(ax2,5:10,means_e,color=:red,linewidth=4)
errorbars!(ax2,5:10,means_e, max.(means_e.-stds_e,means_e),stds_e, color = :black, whiskerwidth = 10,linewidth=3)

fig