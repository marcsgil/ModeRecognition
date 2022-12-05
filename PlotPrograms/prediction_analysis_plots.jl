using LinearAlgebra,Statistics,JLD2,CairoMakie,Flux,DataFrames,CSV

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

function error(ŷ,y)
    sum(abs,y.-ŷ,dims=1)./size(y,1) |> vec
end
##
N_photons = 1024

best_model = load("WeakBeam/Results/Order1/$(N_photons)Photons/AdamW+MAE/best_model.jld2")["model"];
x = load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/xtest_$(N_photons)_photons.jld2")["xtest"]
y = load("C:/MLDatasets/StructuredLight/WeakBeam/Order1/ytest_$(N_photons)_photons.jld2")["ytest"]

ŷ = best_model(x)
##
fs = fidelity(ŷ,y)
mean_f =  round(mean(fs),sigdigits=3)
std_f = round(std(fs),sigdigits=2)

sum(x->x<mean_f ? x : 0,fs)/length(fs)
sum(x->x>mean_f ? x : 0,fs)/length(fs)


errors = error(ŷ,y)
mean_e = round(mean(errors),sigdigits=2)
std_e = round(std(errors),sigdigits=2)
##
fig = Figure(resolution=(1200,600),fontsize=24)
ax1 = CairoMakie.Axis(fig[1,1],xlabel = "Fidelity",title = "Mean: $mean_f;   STD: $std_f")
ax2 = CairoMakie.Axis(fig[1,2],xlabel = "MAE",title = "Mean: $mean_e;   STD: $std_e")
hist!(ax1,fs,bins=LinRange(.99,1,200))
hist!(ax2,errors,bins=LinRange(0,.1,100))
Label(fig[0, :], text = "$(N_photons) Photons",textsize=32)

fig
##
a = round.(y[1,1:5],sigdigits=3)
â = round.(ŷ[1,1:5],sigdigits=3)
b = round.(y[2,1:5],sigdigits=3)
b̂ = round.(ŷ[2,1:5],sigdigits=3)

ex_fid = round.(fidelity(ŷ[:,1:5],y[:,1:5]),digits=4)

df = DataFrame(a2 = a, â2 = â,b2 = b, b̂2 = b̂,Fidelity=ex_fid,)
CSV.write("WeakBeam/Results/Order1/$(N_photons)Photons/AdamW+MAE/exemple_output.csv", df)