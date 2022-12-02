using JLD2, Flux

include("../training_pipeline.jl")
include("../vizualization.jl")

function LeNet5(x,y)
    #Creates a LeNet5 network with the correct sizes to work with input x and output y

    height,widht,channels = size(x)
    nclasses = size(y,1)
    out_conv_size = (height÷4 - 3, widht÷4 - 3, 16)
    
    return Chain(
            Conv((5, 5), channels=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            Flux.flatten,
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses)
          )
end
##
xtrain,xtest = Flux.splitobs(load("C:/MLDatasets/StructuredLight/IntenseBeam/Order1/x_order_1.jld2")["x"], at=0.85);
ytrain,ytest = Flux.splitobs(load("C:/MLDatasets/StructuredLight/IntenseBeam/Order1/y_order_1.jld2")["y"], at=0.85);
##
network = LeNet5(xtrain,ytrain) |> gpu;
train_network!(network, (xtrain,ytrain),(xtest,ytest),saving_path="IntenseBeam/Results/Order1",epochs=200)
##
best_model = load("IntenseBeam/Results/Order1/model.jld2")["model"] |>gpu;
N_photons = 1024
data = load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/x_$(N_photons)_photons.jld2")["x"],
load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/y_$(N_photons)_photons.jld2")["y"]
get_metrics(Flux.DataLoader(data, batchsize=256),best_model,Flux.mse)
view(data[2],:,200:204)
best_model(data[1][:,:,:,200:204]|>gpu)

using LinearAlgebra
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
        cs[1,n] = √(1-sum(abs2,@view cs[2:end,n]))
        ĉs[1,n] = √(1-sum(abs2,@view ĉs[2:end,n]))
    end
    [abs2(dot(view(cs,:,n),view(ĉs,:,n))) for n in axes(cs,2)]
end

slow_fidelity(best_model(data[1][:,:,:,200:204]|>gpu)|>cpu,view(data[2],:,200:204))