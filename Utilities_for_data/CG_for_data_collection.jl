include("Struct_sub_graph.jl")
include("Label_algo_optimized.jl")
include("ReadLogs.jl")
include("Structs.jl")
include("Parallel_Solve.jl")
include("Setup_Master.jl")
include("dataload_structs.jl")
include("Choice_models.jl")
using CSV
using BenchmarkTools

function main_o_all(demand_files::Vector{String}, instance_files::Vector{String}, output_file::String, static::Bool)
    
    total_times = Vector{Float64}()
    obs_vector = Vector{Observations}()
    total_paths_vec = Vector{Int}()
    #all_paths = Vector{Path}()
    #list_of_sub_graphs = []
    obj_val = Vector{Float64}()
    #master = 0
    G = Graph(Dict{String, Vector{Node}}(), Dict{String, Node}(), Dict{Tuple{Int64, Int64}, Edge}(),Vector{Edge}(), Vector{Node}(),Vector{Vector{Node}}(), 0, Vector{Int64}(), 0,0, Vector{Node}(), 0, Dict{Tuple{String,String}, Vector{Int}}(), Dict{Tuple{String,String}, Float64}(), 0, 0)
    total_times_sub = Vector{Float64}()
    for d in 1:length(demand_files)
        total_paths = 0

        env = create_env(10000.0,2.0,1.0, demand_files[d], instance_files[d], 100.0, 1.0)
        G = Graph(Dict{String, Vector{Node}}(), Dict{String, Node}(), Dict{Tuple{Int64, Int64}, Edge}(),Vector{Edge}(), Vector{Node}(),Vector{Vector{Node}}(), 0, Vector{Int64}(), 0,0, Vector{Node}(), 0, Dict{Tuple{String,String}, Vector{Int}}(), Dict{Tuple{String,String}, Float64}(), 0, 0)
        G.build_star_graph(env)

        list_of_sub_graphs = create_sub_problems_o_all(env, G)
        path_list_master, voy_edges_cap_list =  init_forfeit_paths(env, G)

        total_time = time()

        master, x, edges_cons, demand_cons = setup_master(Gurobi.Optimizer,path_list_master, voy_edges_cap_list, env.demand_list)
        optimize!(master)
        pis = dual.(demand_cons)
        lambdas = dual.(edges_cons)
        G.update_lambda(lambdas)
        G.update_pi(pis)


        t1 = time()
        list_of_path_lists,lort,lort = solve_sub_problems_oall(list_of_sub_graphs, G,  0.0, no_choice, 16)
        elapsed_time = time() - t1
        push!(total_times_sub, elapsed_time)
        println("time to solve sub problems ", elapsed_time)
        if !static
            for sub in list_of_sub_graphs
                if sum(sub.data_frame.in_path) > 0
                    new = deepcopy(sub.data_frame)
                    new.normalize_dyn(sub)
                    push!(obs_vector, new)
                end
            end
        end



        path_list = syncronize_all(list_of_path_lists)
        #append!(all_paths, path_list)
        append!(path_list_master, path_list)

        iter = 1
        while length(path_list)>0
            println(length(path_list))
            total_paths += length(path_list)
            t1 = time()
            add_paths_to_master(path_list, env.demand_list, master, edges_cons, demand_cons, x,0.0)
            elapsed_time = time() - t1
            println("Time to add paths to master ", elapsed_time)

            t1 = time()
            optimize!(master)
            #FIND OPTIMAL FOR EXPLANATORY VARIABLE 
            if static
                for var in 1:length(x)
                    if value(x[var])> (0.0 + 0.000001)
                        path = path_list_master[var]
                        sub_id = path.sub_id
                        if sub_id != 0
                            for edge_id in path.sub_edges
                                if edge_id != 0
                                    edge = list_of_sub_graphs[sub_id].edges_ordered[edge_id]
                                    if edge.type in ["voyage_edge", "transhipment_edge"]
                                        list_of_sub_graphs[sub_id].data_frame.in_opt[edge.obs_id] += 1
                                    end
                                end
                            end
                        end
                        
                    end
                end

            end

            elapsed_time = time() - t1
            println("Time to optimize master ", elapsed_time)


            pis = dual.(demand_cons)
            lambdas = dual.(edges_cons)
            G.update_lambda(lambdas)
            G.update_pi(pis)

            t1 = time()
            list_of_path_lists,lort,lort = solve_sub_problems_oall(list_of_sub_graphs, G, 0.0, no_choice, 16)
            elapsed_time = time() - t1
            push!(total_times_sub, elapsed_time)
            if !static
                if length(list_of_path_lists) > 0
                    for sub in list_of_sub_graphs
                        println(sub.id)
                        if sum(sub.data_frame.in_path) > 0
                            new = deepcopy(sub.data_frame)
                            new.normalize_dyn(sub)
                            push!(obs_vector, new)                    
                        end                
                    end
                end
            end
            println("time to solve sub problems ", elapsed_time)
            path_list = syncronize_all(list_of_path_lists)
            #append!(all_paths, path_list)
            append!(path_list_master, path_list)

            iter +=1
        end
        if static 
            for sub in list_of_sub_graphs
                # use this if inopt
                #new = deepcopy(sub.data_frame)
                #if sum(sub.data_frame.in_opt) > 0
                #    new.in_path = new.in_opt.>0
                #    push!(obs_vector, new)                    
                #end 
                #
                ##### use this if inpath
                new = deepcopy(sub.data_frame)
                new.in_path = new.times_chosen_for_sub.> 0
                push!(obs_vector, new)          
            end
        end
        push!(obj_val, objective_value(master))
        push!(total_times, time()- total_time)
        push!(total_paths_vec, total_paths)
    end
    println("Obs to dataframe")
    df = struct_to_dataframe(obs_vector)
    println("Saving Dataframe with size: ", size(df))
    CSV.write(output_file*".csv", df)
    println("total times were ", total_times)
    println("total paths ", total_paths_vec)
    println("Total times for subproblems was ", total_times_sub)
    println("obj_val was ", obj_val)
    println("total_sub was ", sum(total_times_sub))
    println("Ended")
    return df
end
ea_demands = ["Demand_EuropeAsia"  ,  "Demand_EuropeAsia" , "Demand_EuropeAsia" ] 
ea_instances = ["EuropeAsia_pid_26674_12", "EuropeAsia_pid_824_4_best_high","EuropeAsia_pid_26999_11_low_best"]
pa_demands = ["Demand_Pacific",  "Demand_Pacific",  "Demand_Pacific"] 
pa_instances = [ "Pacific_base_best","Pacific_high_best_30600_4","Pacific_28043_5_low_best"]

all_instances = vcat(["Instance-WL.txt"])
all_demands = vcat(["Demand_WorldLarge"])


for (i,instance) in enumerate(all_instances)
    main_o_all([all_demands[i]],[instance],"Data_inpath_medsidstesub/"*instance*"_dynamic",false)
end

for (i,instance) in enumerate(all_instances)
    main_o_all([all_demands[i]],[instance],"Data_inpath_medsidstesub_2/"*instance*"_static",true)
end

