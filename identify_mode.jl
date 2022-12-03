using JLD2, Flux

include("training_pipeline.jl")
include("vizualization.jl")

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
N_photons = 32
xtrain,xtest = Flux.splitobs(load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/x_$(N_photons)_photons.jld2")["x"], at=0.85);
ytrain,ytest = Flux.splitobs(load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/y_$(N_photons)_photons.jld2")["y"], at=0.85);
##
network = LeNet5(xtrain,ytrain) |> gpu;
train_network!(network, (xtrain,ytrain),(xtest,ytest),saving_path="WeakBeam/Results/Order1/$(N_photons)Photons/AMSGrad+MAE",epochs=100,optimizer=AMSGrad(),batchsize=512,loss=Flux.mae)
##
best_model = load("WeakBeam/Results/Order1/32Photons/AdamW/best_model.jld2")["model"] |>gpu;

data = load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/x_$(N_photons)_photons.jld2")["x"],
load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/y_$(N_photons)_photons.jld2")["y"]
get_metrics(Flux.DataLoader((xtest,ytest), batchsize=256),best_model,Flux.mse)
interval = 1:2048
view(ytest,:,interval)
best_model(xtest[:,:,:,interval]|>gpu)
Flux.mae(ytest[:,interval]|>gpu,best_model(xtest[:,:,:,interval] |> gpu))