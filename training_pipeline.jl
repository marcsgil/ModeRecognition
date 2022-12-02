using Flux,ProgressMeter,CSV,DataFrames,JLD2

modified_sqrt(x) = √ifelse(x>0,x,zero(x))

f(x,y) = 1-x^2-y^2

function fidelity(ŷ,y)
    D = size(y,1)÷2
    r̂ = @view ŷ[1:D,:]
    î = @view ŷ[D+1:end,:]
    r = @view y[1:D,:]
    i = @view y[D+1:end,:]

    ĉ1 = modified_sqrt.(sum(map(f,r̂,î),dims=1))
    c1 = modified_sqrt.(sum(map(f,r,i),dims=1))

    R = sum((@. r̂*r+î*i), dims = 1) + ĉ1.*c1
    I = sum((@. r*î-r̂*i), dims = 1)

    sum( @. (R^2+I^2)/( (ĉ1^2+r̂^2+î^2)*(c1^2+r^2+i^2) ) )/size(y,2)
end

function get_metrics(loader,network,loss)
    l = 0f0
    f = 0f0
    ntot = 0
    for (x, y) in loader
        x, y = x |> gpu, y |> gpu
        l += loss(network(x), y) * size(x)[end]     
        f += fidelity(network(x), y) * size(x)[end]   
        ntot += size(x)[end]
    end
    (l/ntot),(f/ntot)
end

function save_metrics(file_path,loss,fidelity)
    if isfile(file_path)
        CSV.write(file_path, DataFrame(loss=loss,fidelity=fidelity),append=true)
    else
        CSV.write(file_path, DataFrame(loss=loss,fidelity=fidelity))
    end
end

function train_network!(network, train_data,test_data;saving_path,
    loss = Flux.mse, optimizer= Adam(), batchsize=256,epochs=5,report=default_report)

    train_loader = Flux.DataLoader(train_data, batchsize=batchsize, shuffle=true)
    test_loader = Flux.DataLoader(test_data, batchsize=batchsize)

    parameters =  Flux.params(network)

    minimum_loss = Inf

    for epoch in 1:epochs
        @showprogress for data in train_loader
            data = data |> gpu
            Flux.train!((x,y)->loss(network(x),y), parameters,[data], optimizer)
        end
        @info "Epoch $epoch"

        println("Train Data")
        l,f = get_metrics(train_loader,network,loss)
        println("Loss: $l; Fidelity: $f")
        save_metrics(saving_path*"/train.csv",l,f)
        println("-------------------------")
        
        println("Test Data")
        l,f = get_metrics(test_loader,network,loss)
        println("Loss: $l; Fidelity: $f")
        save_metrics(saving_path*"/test.csv",l,f)
        if l < minimum_loss
            minimum_loss = l
            jldsave(saving_path*"/best_model.jld2"; model = network |> cpu)
        end
    end
    
    jldsave(saving_path*"/last_model.jld2"; model = network |> cpu)
end