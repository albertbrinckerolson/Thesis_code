include("Struct_sub_graph.jl")
include("Label_algo_optimized.jl")
include("ReadLogs.jl")
include("Structs.jl")
include("Parallel_Solve.jl")
include("Setup_Master.jl")
include("dataload_structs.jl")

using MLJ
using ScikitLearn
using MLJScikitLearnInterface
using PyCall
using JLD2
using CategoricalArrays
using BenchmarkTools
using Statistics
using BSON: @save,@load
using Flux, CUDA, Statistics, ProgressMeter

#env = create_env(10000.0,2.0,1.0, "Demand_Pacific", "Pacific_base_best")
env = create_env(10000.0,2.0,1.0, "Demand_WorldLarge","WorldLarge_base.txt")
G = Graph(Dict{String, Vector{Node}}(), Dict{String, Node}(), Dict{Tuple{Int64, Int64}, Edge}(),Vector{Edge}(), Vector{Node}(),Vector{Vector{Node}}(), 0, Vector{Int64}(), 0,0, Vector{Node}(), 0, Dict{Tuple{String,String}, Demand}(), Dict{Tuple{String,String}, Float64}(), 0, 0)

G.build_star_graph(env)
list_of_sub_graphs = create_sub_problems_o_all(env, G)
test_sub = list_of_sub_graphs[1]

test_sub.data_frame
round(maximum(obs_to_floats(test_sub.data_frame)), digits = 5)

df = obs_to_floats(test_sub.data_frame)
test_sub.data_frame.normalize(test_sub)

obs_to_floats(test_sub.data_frame)

@load "Nye_Modeller_inpath/WorldLarge_base.txt_static.csv.bson" model
preds =  softmax(model(obs_to_floats(test_sub.data_frame))).> 0.5
y_hat = preds[1,:]



function df_2_dataset(filenames,edges::Vector{String}, static::Bool, oall::Bool,return_df::Bool)
    Xs = []
    Ys = []
    dfs = []
    Ws = []
    to_remove_after = ["instance_name","edge_id","sub_graph_id","in_path","iteration","in_opt"]
    to_remove = ["load_edge","forfeit_edge","sink_edge","times_chosen_for_sub"]
    use_voy = true
    use_trans = true
    if !("voy_edge" in edges)
        use_voy = false
    end 
    if !("trans_edge" in edges)
        use_trans = false
    end 
    if !(use_voy & use_trans)
        push!(to_remove_after,"trans_edge")
        push!(to_remove_after,"voy_edge")
    end   
    name = nothing
    for filename in filenames
        df = CSV.read(filename, DataFrame, ntasks=16)

        # Filter col names for removal
        for name in names(df)
            if contains(name,"base")
                push!(to_remove,name)
            end
            if contains(name,"max")
                push!(to_remove,name)
            end

            if contains(name,"pi")
                push!(to_remove,name)
            end
            if contains(name,"pj")
                push!(to_remove,name)
            end
            if static
                if contains(name,"lambda") 
                     push!(to_remove,name)
                end
            end
            if oall 
                if contains(name,"dist_less")
                    push!(to_remove,name)  
                end
                if contains(name,"dest")
                    push!(to_remove,name)  
                end
            end
        end
        df = filter(row->row.iteration != 1,df)
        # drop unused collumns
        df = filter(row->row.load_edge != 1,df)
        df = filter(row->row.sink_edge != 1,df)
        df = filter(row->row.forfeit_edge != 1,df)
        if !use_trans
            df = filter(row->row.trans_edge != 1,df)
        end
        if !use_voy
            df = filter(row->row.voy_edge != 1,df)
        end

        # extract target
        Y = df[!,"in_path"]
        W = df[!,"in_opt"]
        W = [(1+w) for w in W]
        class_bal = sum(Y)/length(Y)

        # remove unused collumns
        select!(df, Not([Symbol(rem) for rem in to_remove]))
        push!(dfs,deepcopy(df))
        select!(df, Not([Symbol(rem) for rem in to_remove_after]))
        println("Using cols $(names(df))")
        push!(Xs,deepcopy(transpose(Matrix(df))))
        push!(Ys,Y)
        push!(Ws,W)

    end
    if length(filenames) > 1
        X = hcat(Xs...)
        DF = vcat(dfs...)
        Y = vcat(Ys...)
        W = vcat(Ws...)
    else
        Y = Ys[1]
        X = Xs[1]
        W = Ws[1]
        DF = dfs[1]
    end
    ### add model prediction probabilities

    class_bal = sum(Y)/length(Y)
    if return_df
        return X,Y,W,class_bal,DF
    else
        return X,Y,W,class_bal
    end
end
static = true
edges = ["voy_edge","trans_edge"]
x_train, y_train, w_train, pos_class_fraq_train,df = df_2_dataset(["Ny_Data_Folder_inopt/Instance-WL.txt_static.csv"],edges,static,true,true)
x_train = Float32.(x_train)
df_sub1 = filter(row -> row.sub_graph_id == 1, df)

obs_to_floats(test_sub.data_frame)


lol = obs_to_floats(test_sub.data_frame)
y_pred = model(lol) # Prediction
probs = softmax(y_pred)
for prob in probs
    println(prob)
end
test_sub.predict(model)

MLJ.fit!(model,force = true)
MLJ.report(model)

Threads.@threads for i in 1:2
    test_sub.predict(model)
end

println(model.model.nr_threads)


model = "RF_models_inopt_2/WorldLarge_base.txt_static.csv_RF.jlso"

model = machine(model)
typeof(model)
for edge in test_sub.edges_ordered
    println(edge.prob)
end
env1 = create_env(10000.0,2.0,1.0, "Demand_EuropeAsia","EuropeAsia_pid_26674_12")
G1 = Graph(Dict{String, Vector{Node}}(), Dict{String, Node}(), Dict{Tuple{Int64, Int64}, Edge}(),Vector{Edge}(), Vector{Node}(),Vector{Vector{Node}}(), 0, Vector{Int64}(), 0,0, Vector{Node}(), 0, Dict{Tuple{String,String}, Vector{Int}}(), Dict{Tuple{String,String}, Float64}(), 0, 0)
G1.build_star_graph(env1)

env = create_env(10000.0,2.0,1.0, "Demand_WorldLarge","Instance-WL.txt", 100.0, 1.0)
G = Graph(Dict{String, Vector{Node}}(), Dict{String, Node}(), Dict{Tuple{Int64, Int64}, Edge}(),Vector{Edge}(), Vector{Node}(),Vector{Vector{Node}}(), 0, Vector{Int64}(), 0,0, Vector{Node}(), 0, Dict{Tuple{String,String}, Vector{Int}}(), Dict{Tuple{String,String}, Float64}(), 0, 0)
G.build_star_graph(env)
list_of_sub_graphs = create_sub_problems_o_all(env, G)








path_list_master1, voy_edges_cap_list1 =  init_forfeit_paths(env1, G1)
path_list_master, voy_edges_cap_list =  init_forfeit_paths(env, G)

path_demand_ids = Set()
for path in path_list_master
    push!(path_demand_ids, path.demand_id)
end
demand_demand_ids = Set()
for demand in env.demand_list
    push!(demand_demand_ids, demand.id)
end
"""
"""
demand_demand_ids
path_demand_ids
length(path_list_master)
length(demand_demand_ids)

for (i, demand) in enumerate(env.demand_list)
    for (j,demand2) in enumerate(env.demand_list[i+1:end])
        if ((demand.origin == demand2.origin) & (demand.destination == demand2.destination))
            println(demand, demand2)
        end
    end
end


for (i, path) in enumerate(path_list_master)
    for (j, path2) in enumerate(path_list_master[i+1:end])
        if path.demand_id == path2.demand_id 
            println("BIG MISTAKE")
        end
    end
end

master1, x1, edges_cons1, demand_cons1 = setup_master(Gurobi.Optimizer, path_list_master1, voy_edges_cap_list1, env1.demand_list)
master, x, edges_cons, demand_cons = setup_master(Gurobi.Optimizer, path_list_master, voy_edges_cap_list, env.demand_list)

optimize!(master)
pis = dual.(demand_cons)
lambdas = dual.(edges_cons)
G.update_lambda(lambdas)
G.update_pi(pis)
list_of_sub_graphs[1].update_dual(G)
list_of_sub_graphs[1].data_frame.arcs_lambda_j_sub_max




include("Choice_models.jl")
list_of_path_lists = solve_sub_problems(list_of_sub_graphs, G, 0.0, no_choice, 16)
path_list = syncronize_all(list_of_path_lists)
iter = 1
println(iter)
println(length(path_list))

while (length(path_list)>0) & (iter < 10)
    add_paths_to_master(path_list, env.demand_list, master, edges_cons, demand_cons, x, 0)
    optimize!(master)
    pis = dual.(demand_cons)
    lambdas = dual.(edges_cons)
    G.update_lambda(lambdas)
    G.update_pi(pis)
    list_of_path_lists = solve_sub_problems(list_of_sub_graphs, G, 16)
    path_list = syncronize_all(list_of_path_lists)
    iter += 1
    println(iter)
    println(length(path_list))
end
end
main()

function read_demand_all(path::String)
    demand_df = CSV.read(path, DataFrame, ntasks=1)
    #o_all = Dict(collect(ports)[i] => Vector{Demand}() for i in 1:length(ports))
    id = 1
    demand_list = Vector{Demand}()
    for row in eachrow(demand_df)
        #if haskey(o_all,row.Origin) 
            #if row.Destination in ports
                new_demand = Demand(id,row.Origin, row.Destination, row.FFEPerWeek, row.Revenue_1, row.TransitTime+100)
                push!(demand_list, new_demand)
                id+=1
            #end
        #end

    end
    return demand_list
end
demand_list = read_demand("C:/Users/peder/OneDrive/Dokumenter/GitHub/Thesis-Column-Generation-2024/LINERLIB-master/data/Demand_Baltic.csv")
total = 0
for demand in demand_list
    total += demand.ffe_per_week
end