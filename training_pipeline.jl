using Flux,ProgressMeter,CSV,DataFrames,JLD2

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

    sum( @. (R^2+I^2)/( N*N̂ ) )/size(y,2)
end

fidelity_loss(ŷ,y) = - fidelity(ŷ,y)

function get_metrics(loader,network,loss)
    l = 0f0
    ntot = 0
    for (x, y) in loader
        x, y = x |> gpu, y |> gpu
        l += loss(network(x|> gpu), y|> gpu) * size(x)[end]     
        ntot += size(x)[end]
    end
    (l/ntot)
end

function save_metrics(file_path,loss)
    if isfile(file_path)
        CSV.write(file_path, DataFrame(loss=loss),append=true)
    else
        CSV.write(file_path, DataFrame(loss=loss))
    end
end

function train_network!(network, train_data,test_data;saving_path,
    loss = Flux.mse, optimizer= Adam(), batchsize=256, epochs=5, stop=5)

    train_loader = Flux.DataLoader(train_data, batchsize=batchsize, shuffle=true)
    test_loader = Flux.DataLoader(test_data, batchsize=batchsize)

    parameters =  Flux.params(network)

    minimum_loss = Inf
    minimum_loss_epoch = 0

    for epoch in 1:epochs
        @showprogress for data in train_loader
            data = data |> gpu
            Flux.train!((x,y)->loss(network(x),y), parameters,[data], optimizer)
        end
        @info "Epoch $epoch"

        println("Train Data")
        l = get_metrics(train_loader,network,loss)
        println("Fidelity: $(-l)")
        save_metrics(saving_path*"/train.csv",-l)
        println("-------------------------")
        
        println("Test Data")
        l = get_metrics(test_loader,network,loss)
        println("Fidelity: $(-l)")
        save_metrics(saving_path*"/test.csv",-l)
        if l < minimum_loss
            minimum_loss = l
            minimum_loss_epoch = epoch
            jldsave(saving_path*"/best_model.jld2"; model = network |> cpu)
        end

        if epoch-minimum_loss_epoch ≥ stop
            break
        end
    end
    
    jldsave(saving_path*"/last_model.jld2"; model = network |> cpu)
end