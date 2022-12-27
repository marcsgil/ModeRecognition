using Symbolics
using CUDA
CUDA.allowscalar(false)

function get_function_v1(D)
    if D==1
        @variables x₁
        build_function( prod(x₁),x₁) |> eval
    elseif D == 2
        @variables x₁ x₂
        build_function( prod((x₁,x₂)),x₁,x₂) |> eval
    elseif D == 3
        @variables x₁ x₂ x₃
        build_function( prod((x₁,x₂,x₃)),x₁,x₂,x₃) |> eval
    end
end

function get_function_v2(D)
    @variables x[1:D]

    build_function( prod(x),ntuple(i->x[i],D)...) |> eval
end

f1 = get_function_v1(3)
f2 = get_function_v2(3)

f2(1,1,1)

map(f1,CUDA.rand(10),CUDA.rand(10),CUDA.rand(10))
map(f1,CUDA.rand(10),CUDA.rand(10),CUDA.rand(10))
##

assign = "x = 1"
ex1 = Meta.parse(assign)
typeof(ex1)
ex1.args
##
ntuple(n->(@variables Symbol("x","$n"))),4
Symbol("x","1")##
exp = Meta.parse("@variables"*prod(n->" x$n",1:10))
eval(exp)
typeof(x1)
typeof(Meta.parse("x1")|>eval)