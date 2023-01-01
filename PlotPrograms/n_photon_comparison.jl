using CairoMakie,JLD2,Flux,LinearAlgebra,Statistics,ProgressMeter,ColorSchemes

function fidelity(ŷ,y)
    D = size(y,1)÷2
    r̂ = @view ŷ[2:D+1,:]
    î = @view ŷ[D+2:end,:]
    r = @view y[2:D+1,:]
    i = @view y[D+2:end,:]

    ĉ1 = @view ŷ[1:1,:]
    c1 = @view y[1:1,:]

    R = sum((@. r̂*r+î*i), dims = 1) .+ ĉ1.*c1
    I = sum((@. r*î-r̂*i), dims = 1)

    N = sum(abs2,r,dims=1) + sum(abs2,i,dims=1) + c1.^2
    N̂ = sum(abs2,r̂,dims=1) + sum(abs2,î,dims=1) + ĉ1.^2

    @. (R^2+I^2)/( N*N̂ )/size(y,2)
end
##
N_photons = [32,64,128,256,512,1024,2048,Inf]
orders = [1,2,3,4]
pars = Iterators.product(N_photons,orders) |> collect

means_f = Array{Float32}(undef,length(N_photons),length(orders))
stds_f = Array{Float32}(undef,length(N_photons),length(orders))
##
@showprogress for n in eachindex(pars)
    N_photon,order = pars[n]
    if !isinf(N_photon)
        N_photon = Int(N_photon)
    end
    order = Int(order)

    best_model = load("Theoretical/Order$(order)/$(N_photon)_photons/best_model.jld2")["model"];
    xtrain,xtest = Flux.splitobs(load("C:/MLDatasets/StructuredLight/Theoretical/x_order$(order)_$(N_photon)_photons.jld2")["x"], at=0.85);
    ytrain,ytest = Flux.splitobs(load("C:/MLDatasets/StructuredLight/Theoretical/y_order$(order)_$(N_photon)_photons.jld2")["cs"], at=0.85);

    xtrain = nothing
    ytrain = nothing

    ŷ = best_model(xtest)
    fs = fidelity(ŷ,ytest)

    means_f[n] =  mean(fs)
    stds_f[n] = std(fs)
end
jldsave("Theoretical/fidelities.jld2",means=means_f,stds=stds_f)
##
#means_f = load("Theoretical/fidelities.jld2")["means"]
#stds_f = load("Theoretical/fidelities.jld2")["stds"]
stds_f = round.(stds_f,sigdigits=2)
function first_sigdigit(x)
    @assert x<1
    for n in 1:100
        if x*10^n ≥ 1
            return n
            break
        end
    end
end

for n in eachindex(means_f)
    means_f[n] = round(means_f[n],digits=first_sigdigit(stds_f[n])+1)
end

means_f
##
xlabels = [string(2^x) for x in 5:11]
push!(xlabels,"Inf")


fig = Figure(fontsize=28)
ax = Axis(fig[1,1],xlabel = "Number of photons", ylabel = "Mean Fidelity",xticks=(5:12,xlabels),yticks=.94:.01:1.05)
ylims!(ax,0.94,1.001)
#xlims!(5,10)
for n in axes(means_f,2)
    lines!(ax,5:12, means_f[:,n],linewidth=4,label="Dimension $(n+1)")
    #errorbars!(ax,5:12, means_f[:,n],stds_f[:,n], whiskerwidth = 10,linewidth=3)
end
axislegend(position = :rb)

fig
##
stds_f
means_f
##
means_f
fig,ax,hm = heatmap(2:5,5:12,means_f',axis=(;yticks=(5:12,xlabels),xticks=(2:5),xlabel="Dimension",ylabel="Number of Photons"),figure=(;fontsize=24,resolution=(1000,600)),colorrange=(.94,1.00),colormap=:balance)
Colorbar(fig[1,2],hm,ticks=.94:.01:1,label="Fidelity")
for i in 2:5
    for j in 5:12
        textcolor = means_f[j-4,i-1] > .96 &&  means_f[j-4,i-1] < .997 ? :black : :white
        text!(ax, string(means_f[j-4,i-1])*"±"*string(stds_f[j-4,i-1]), position = (i, j),
        color = textcolor, align = (:center, :center))
    end
end
fig
##

ColorSchemes.balance[0.1]