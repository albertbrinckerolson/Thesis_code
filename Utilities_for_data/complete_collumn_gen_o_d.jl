include("Struct_sub_graph.jl")
include("Label_algo_optimized.jl")
include("ReadLogs.jl")
include("Choice_models.jl")
include("Structs.jl")
include("Parallel_Solve.jl")
include("Setup_Master.jl")
include("dataload_structs.jl")
using CSV
using BenchmarkTools

function main_od(demand_files::Vector{String}, instance_files::Vector{String}, output_file, static::Bool)
    
    total_times = Vector{Float64}()
    obs_vector = Vector{Observations}()
    #total_paths_vec = Vector{Int}()
    #all_paths = Vector{Path}()
    #G = Graph(Dict{String, Vector{Node}}(), Dict{String, Node}(), Dict{Tuple{Int64, Int64}, Edge}(),Vector{Edge}(), Vector{Node}(),Vector{Vector{Node}}(), 0, Vector{Int64}(), 0,0, Vector{Node}(), 0, Dict{Tuple{String,String}, Vector{Int}}(), Dict{Tuple{String,String}, Float64}(), 0, 0)
    #test_sub = 0
    #list_of_sub_graphs = []
    #master = 0
    for d in 1:length(demand_files)
        total_paths = 0

        env = create_env(10000.0,2.0,1.0, demand_files[d], instance_files[d], 100.0, 1.0)
        G = Graph(Dict{String, Vector{Node}}(), Dict{String, Node}(), Dict{Tuple{Int64, Int64}, Edge}(),Vector{Edge}(), Vector{Node}(),Vector{Vector{Node}}(), 0, Vector{Int64}(), 0,0, Vector{Node}(), 0, Dict{Tuple{String,String}, Vector{Int}}(), Dict{Tuple{String,String}, Float64}(), 0, 0)
        G.build_star_graph(env)

        list_of_sub_graphs = create_sub_problems(env, G)
        path_list_master, voy_edges_cap_list =  init_forfeit_paths(env, G)
        #test_sub = list_of_sub_graphs[1]
        total_time = time()

        master, x, edges_cons, demand_cons = setup_master(Gurobi.Optimizer, path_list_master, voy_edges_cap_list, env.demand_list)
        optimize!(master)
        pis = dual.(demand_cons)
        lambdas = dual.(edges_cons)
        G.update_lambda(lambdas)
        G.update_pi(pis)


        t1 = time()
        list_of_path_lists = solve_sub_problems(list_of_sub_graphs, G, 0.0, test_choice, 16)

        elapsed_time = time() - t1
        println("time to solve sub problems ", elapsed_time)

        path_list = syncronize_all(list_of_path_lists)
        #append!(all_paths, path_list)

        iter = 1
        while (length(path_list)>0)
            #println(length(path_list))
            #total_paths += length(path_list)
            t1 = time()
            add_paths_to_master(path_list, env.demand_list, master, edges_cons, demand_cons, x, 0)
            elapsed_time = time() - t1
            println("Time to add paths to master ", elapsed_time)

            t1 = time()
            optimize!(master)
            elapsed_time = time() - t1
            println("Time to optimize master ", elapsed_time)


            pis = dual.(demand_cons)
            lambdas = dual.(edges_cons)
            G.update_lambda(lambdas)
            G.update_pi(pis)

            t1 = time()
            list_of_path_lists = solve_sub_problems(list_of_sub_graphs, G, 0.0, test_choice, 16)

            elapsed_time = time() - t1
            println("time to solve sub problems ", elapsed_time)
            path_list = syncronize_all(list_of_path_lists)
            #append!(all_paths, path_list)

            iter +=1
        end
        push!(total_times, time()- total_time)
        #push!(total_paths_vec, total_paths)




    end

    println("Obs to dataframe")
    #df = struct_to_dataframe(obs_vector)
    println("Saving Dataframe with size: ", size(df))
    #CSV.write(output_file*".csv", df)
    println("total times were ", total_times)
    #println("total paths ", total_paths_vec)
    println("Ended")
    return 
end

train_demands = ["Demand_EuropeAsia" , "Demand_Mediterranean", "Demand_Pacific","Demand_EuropeAsia" , "Demand_Mediterranean", "Demand_Pacific","Demand_EuropeAsia" , "Demand_Mediterranean", "Demand_Pacific"] 
train_instances = ["EuropeAsia_pid_26674_12", "Med_base_best", "Pacific_base_best","EuropeAsia_pid_824_4_best_high","Med_high_best_5840_1","Pacific_high_best_30600_4","EuropeAsia_pid_26999_11_low_best","Med_low_best_12808_11","Pacific_28043_5_low_best"]
test_demands = ["Demand_WorldSmall_Fixed_Sep","Demand_WorldSmall_Fixed_Sep","Demand_WorldSmall_Fixed_Sep"]
test_instances = ["WorldSmall_Best_Base","WorldSmall_high_best_12848_1","WorldSmall_pid_4953_15_best_low"]

df_train = main_od(train_demands,train_instances, "train_od_BIG_t100_df", true)
df_test = main_od(test_demands,test_instances, "test_od_BIG_t100_df", true)

#df_test = main1(["Demand_WorldSmall_Fixed_Sep"], ["WorldSmall_Best_Base"], "test_df", true)
