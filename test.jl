using CSV, DataFrames

df = DataFrame(A=1,B=1)

CSV.write("data.csv", Tables.table([1,1]),append=true)