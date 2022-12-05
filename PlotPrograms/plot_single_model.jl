using CairoMakie,Flux,JLD2,ProgressMeter

include("../training_pipeline.jl")

best_model = load("IntenseBeam/Results/Order1/model.jld2")["model"] |>gpu;

losses = Vector{Float32}(undef,6)
fidelities = Vector{Float32}(undef,6)

Ns = [2^n for n in 5:10]

@showprogress for n in eachindex(Ns)
    data = ( load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/x_$(Ns[n])_photons.jld2")["x"],
    load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/y_$(Ns[n])_photons.jld2")["y"] )
    losses[n],fidelities[n] = get_metrics(Flux.DataLoader(data, batchsize=256),best_model,Flux.mse)
end
##
fig = Figure(fontsize=24)
ax = Axis(fig[1,1],yticks=0.9:.02:1,xlabel = L"\log_2 (\text{number of photons})", ylabel = "Average Fidelity")
ylims!(ax,0.89,1)
xlims!(5,10)
line = lines!(ax,5:10,fidelities,color=:blue)

fig
##
fig = Figure(fontsize=24)
ax = Axis(fig[1,1],yticks=0.01:0.01:0.06,xlabel = L"\log_2 (\text{number of photons})", ylabel = "Mean Squared Error")
#ylims!(ax,0.89,1)
xlims!(5,10)
line = lines!(ax,5:10,losses,color=:blue)

fig