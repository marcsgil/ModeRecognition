using JLD2, Flux

include("training_pipeline.jl")
include("vizualization.jl")

using CUDA
CUDA.allowscalar(false)

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
order=1
N_photon = 32
if ! isinf(N_photon)
  N_photon = Int(N_photon)
end

xtrain,xtest = Flux.splitobs(load("C:/MLDatasets/StructuredLight/Theoretical/x_order$(order)_$(N_photon)_photons.jld2")["x"], at=0.85);
ytrain,ytest = Flux.splitobs(load("C:/MLDatasets/StructuredLight/Theoretical/y_order$(order)_$(N_photon)_photons.jld2")["cs"], at=0.85);

network = LeNet5(xtrain,ytrain) |> gpu;

try
  mkdir("Theoretical/Order$order/$(N_photon)_photons")
catch
  mkdir("Theoretical/Order$order")
  mkdir("Theoretical/Order$order/$(N_photon)_photons")
end
train_network!(network, (xtrain,ytrain),(xtest,ytest),saving_path="Theoretical/AdamWOrder1_32v2",
epochs=300,loss=fidelity_loss,stop=40,optimizer=AdamW(0.001, (0.9, 0.999), 5e-4))
##
N_photons = [Inf,32,64,128,256,512,1024,2048]
orders = [2,3,4]
pars = Iterators.product(N_photons,orders)

for (N_photon,order) in pars
    if ! isinf(N_photon)
      N_photon = Int(N_photon)
    end

    xtrain,xtest = Flux.splitobs(load("C:/MLDatasets/StructuredLight/Theoretical/x_order$(order)_$(N_photon)_photons.jld2")["x"], at=0.85);
    ytrain,ytest = Flux.splitobs(load("C:/MLDatasets/StructuredLight/Theoretical/y_order$(order)_$(N_photon)_photons.jld2")["cs"], at=0.85);

    network = LeNet5(xtrain,ytrain) |> gpu;

    try
      mkdir("Theoretical/Order$order/$(N_photon)_photons")
    catch
      mkdir("Theoretical/Order$order")
      mkdir("Theoretical/Order$order/$(N_photon)_photons")
    end
    train_network!(network, (xtrain,ytrain),(xtest,ytest),saving_path="Theoretical/Order$order/$(N_photon)_photons",
    epochs=300,loss=fidelity_loss,stop=40)
end
##
function normalize_cols(y)
  result = similar(y)
  for n in axes(y,2)
    result[:,n] = y[:,n]/√sum(abs2,y[:,n])
  end
  result
end
##
order=1
N_photon=Inf
best_model = load("Theoretical/Order$(order)/$(N_photon)_photons/best_model.jld2")["model"] |>gpu;

xtrain,xtest = Flux.splitobs(load("C:/MLDatasets/StructuredLight/Theoretical/x_order$(order)_$(N_photon)_photons.jld2")["x"], at=0.85);
ytrain,ytest = Flux.splitobs(load("C:/MLDatasets/StructuredLight/Theoretical/y_order$(order)_$(N_photon)_photons.jld2")["cs"], at=0.85);
##
interval = 10:15
y = view(ytest,:,interval)
ŷ = best_model(xtest[:,:,:,interval]|>gpu) |>cpu |> normalize_cols