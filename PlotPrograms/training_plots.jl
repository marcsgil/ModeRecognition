using CairoMakie,CSV
##
N_photons = 32
path = "WeakBeam/Results/Order1/$(N_photons)Photons/ADAMW+MAE"

train_loss = CSV.File(open(path*"/train.csv"))["loss"]
test_loss = CSV.File(open(path*"/test.csv"))["loss"]
train_fid = CSV.File(open(path*"/train.csv"))["fidelity"]
test_fid = CSV.File(open(path*"/test.csv"))["fidelity"]
##
fig = Figure(resolution=(1200,800),fontsize=24)
ax1 = CairoMakie.Axis(fig[1,1],xlabel = "Epoch", ylabel = "Fidelity",title = "$(N_photons) Photons")
ax2 = CairoMakie.Axis(fig[2,1],xlabel = "Epoch", ylabel = "MAE")
ylims!(ax1,0.95,.971)
ylims!(ax2,0.089,0.125)
#xlims!(ax,first(eachindex(train)),last(eachindex(train)))
train_line = lines!(ax1,eachindex(train_fid),train_fid,color=:blue)
test_line = lines!(ax1,eachindex(test_fid),test_fid,color=:red)

train_line2 = lines!(ax2,eachindex(train_loss),train_loss,color=:blue)
test_line2 = lines!(ax2,eachindex(test_loss),test_loss,color=:red)
Legend(fig[:,2],[train_line,test_line],["Train Data", "Test Data"])
fig