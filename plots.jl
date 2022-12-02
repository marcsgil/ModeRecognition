using CairoMakie,CSV
##
function get_plot(path)
    train = CSV.File(open(path*"/train.csv"))["fidelity"]
    test = CSV.File(open(path*"/test.csv"))["fidelity"]

    fig = Figure()
    ax = CairoMakie.Axis(fig[1,1],xlabel = "Epoch", ylabel = "Fidelity",
    title = "32 Photons")
    #ylims!(ax,0.94,1)
    train_line = lines!(ax,eachindex(train),train,color=:blue)
    test_line = lines!(ax,eachindex(test),test,color=:red)
    axislegend(ax,[train_line,test_line],["Train Data", "Test Data"],position=:rb)

    fig
end

get_plot("WeakBeam/Results/Order1/32Photons/Adam")
##

exps=5:10
photons = map(x->Int(2^x),exps)

max_train = [ maximum(CSV.File(open("WeakBeam/Results/$(n)_photons/train.csv"))["Value"]) for n in photons ]
max_test = [ maximum(CSV.File(open("WeakBeam/Results/$(n)_photons/test.csv"))["Value"]) for n in photons ]
max_intense_test = maximum(CSV.File(open("IntenseBeam/Results/Order1/test.csv"))["fidelity"])
√minimum(CSV.File(open("IntenseBeam/Results/Order1/test.csv"))["loss"])
max_intense_train = maximum(CSV.File(open("WeakBeam/Results/intense/train.csv"))["Value"])
##
fig = Figure(fontsize=24)
ax = Axis(fig[1,1],yticks=[0.97,0.98,0.99,1],xlabel = L"\log_2 (\text{number of photons})", ylabel = "Maximum Average Fidelity")
ylims!(ax,0.968,1)
xlims!(5,10)
train_line = lines!(ax,exps,max_train,color=:blue)
test_line = lines!(ax,exps,max_test,color=:red)

axislegend(ax,[train_line,test_line],["Train Data", "Test Data"],position=:rb)

fig