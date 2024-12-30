

mutable struct Atomic_counter
    @atomic count :: Int
end

function create_sub_problems_o_all(env::Env, G::Graph)
    list_of_sub_graphs = Vector{GraphSub}()
    id = 0
    for (key, demands) in env.demand_dict
        if length(demands) > 0
            id += 1
            Gs = init_sub_graph_o_all(G, key, env, 10000, id)
            Gs.build_subgraph(G, env)
            push!(list_of_sub_graphs, Gs)
        end
    end
    return list_of_sub_graphs
end



function create_sub_problems(env::Env, G::Graph)
    list_of_sub_graphs = Vector{GraphSub}()
    id = 0
    for (key, demands) in env.demand_dict
        if length(demands) > 0
            for demand in demands
                id += 1
                Gs = init_sub_graph_o_d(G, key, env, 10000, id, demand)
                Gs.build_subgraph(G, env)
                push!(list_of_sub_graphs, Gs)
            end
        end
    end
    return list_of_sub_graphs
end


function get_thread_indexes(nr_threads:: Int, list_length::Int)
    threads_indexes = [[(i-1)*div(list_length, nr_threads )+j for j in 1:div(list_length, nr_threads)]   for i in 1:nr_threads]
    c = 1
    for i in nr_threads*div(list_length , nr_threads)+1:list_length
        push!(threads_indexes[c], i)
        c +=1
    end
    return threads_indexes
end

function thread_call_labeling_algo!(list_of_path_lists::Vector{Vector{Path}},  list_of_sub_graphs::Vector{GraphSub}, thread_indexes::Vector{Int}, G::Graph, threshold, choice_function)
    for sub_graph_id in thread_indexes
        no_paths, dual = list_of_sub_graphs[sub_graph_id].update_dual(G)
        if dual <= 0
          #  println("sub_graph ", sub_graph_id, " was skipped by thread ", Threads.threadid())
            continue
        end
        #println("sub_graph ", sub_graph_id, " by thread ", Threads.threadid())
        #edges_id_list = [Vector{Int}() for i in 1:G.number_of_edges]
        path_list = Vector{Path}()
        semi_stoc_labeling_algorithm_opt!(G, path_list, list_of_sub_graphs[sub_graph_id], threshold, choice_function, dual)
        #list_of_Edges_ID_lists[sub_graph_id] = edges_id_list 
        list_of_path_lists[sub_graph_id] = path_list
    end
end

function solve_sub_problems(list_of_sub_graphs::Vector{GraphSub}, G::Graph,threshold, choice_function, nr_threads = Threads.nthreads())
    countr = Threads.Atomic{Int}(1)
    nr_sub_graphs = length(list_of_sub_graphs)
    list_of_path_lists = [Vector{Path}() for i in 1:nr_sub_graphs]
    thread_indexes = get_thread_indexes(nr_threads, nr_sub_graphs)
    Threads.@threads for i in 1:nr_threads
        while true
            ind = Threads.atomic_add!(countr, 1)
            if ind > length(list_of_sub_graphs)
                break
            end
            thread_call_labeling_algo!(list_of_path_lists, list_of_sub_graphs, [ind], G, threshold, choice_function)
        end
    end
    return list_of_path_lists
end



function syncronize_2_master!(path_list_1::Vector{Path}, path_list_2::Vector{Path}, edges_id_list_1::Vector{Vector{Int}}, edges_id_list_2::Vector{Vector{Int}}, demands_paths::Dict{Tuple{String,String}, Vector{Int}})
    path1_length = length(path_list_1)
    append!(path_list_1, path_list_2)
    for i in 1:length(edges_id_list_1)
        for j in 1:length(edges_id_list_2[i])
            edges_id_list_2[i][j] += path1_length 
        end
        append!(edges_id_list_1[i],edges_id_list_2[i])
    end
    for path in path_list_1[path1_length+1:end]
        path.path_id += path1_length
        push!(demands_paths[(path.origin, path.destination)], path.path_id)
    end
end

function syncronize_all(list_of_path_lists::Vector{Vector{Path}})
    return reduce(vcat, list_of_path_lists)
end



function semi_stoc_thread_call_labeling_algo_oall!(list_of_path_lists::Vector{Vector{Path}},list_of_fallbacks::Vector{Int},list_of_label_algo_calls::Vector{Int},  list_of_sub_graphs::Vector{GraphSub}, thread_indexes::Vector{Int}, G::Graph, threshold::Float64, choice_function::Function, model = nothing, model_dyn = nothing)
    for sub_graph_id in thread_indexes
        #println("sub_graph ", sub_graph_id, " by thread ", Threads.threadid())
        no_paths, max_dual = list_of_sub_graphs[sub_graph_id].update_dual(G)
        if no_paths
            continue
        end        #edges_id_list = [Vector{Int}() for i in 1:G.number_of_edges]
        #if list_of_sub_graphs[sub_graph_id].no_paths_found 
        #    model_dyn = nothing
        #end

        if isnothing(model_dyn)
            
        elseif !list_of_sub_graphs[sub_graph_id].static
            list_of_sub_graphs[sub_graph_id].predict(model_dyn)
#           list_of_sub_graphs[sub_graph_id].no_paths_found = false
        end
        path_list = Vector{Path}()
        if list_of_sub_graphs[sub_graph_id].static & isnothing(model)
            semi_stoc_labeling_algorithm_opt!(G, path_list, list_of_sub_graphs[sub_graph_id], 0.0, choice_function, max_dual)
            list_of_label_algo_calls[sub_graph_id] += 1
        elseif list_of_sub_graphs[sub_graph_id].static & !isnothing(model)
            semi_stoc_labeling_algorithm_opt!(G, path_list, list_of_sub_graphs[sub_graph_id], 0.5, choice_function, max_dual)
            list_of_label_algo_calls[sub_graph_id] += 1
        elseif !list_of_sub_graphs[sub_graph_id].static & isnothing(model_dyn)
            semi_stoc_labeling_algorithm_opt!(G, path_list, list_of_sub_graphs[sub_graph_id], 0.0, choice_function, max_dual)
            list_of_label_algo_calls[sub_graph_id] += 1
        else
            semi_stoc_labeling_algorithm_opt!(G, path_list, list_of_sub_graphs[sub_graph_id], list_of_sub_graphs[sub_graph_id].threshold, choice_function, max_dual)
            list_of_label_algo_calls[sub_graph_id] += 1
        end
        if (length(path_list) < 1)
            #list_of_sub_graphs[sub_graph_id].threshold -= 0.1
            list_of_sub_graphs[sub_graph_id].no_paths_found = true
        else
            list_of_sub_graphs[sub_graph_id].no_paths_found = false
        end

        #if (length(path_list) < 1) & !isnothing(model) & !list_of_sub_graphs[sub_graph_id].no_paths_found
            #list_of_sub_graphs[sub_graph_id].no_paths_found = true
            #semi_stoc_labeling_algorithm_opt!(G, path_list, list_of_sub_graphs[sub_graph_id], 0.0, choice_function, max_dual)
            #list_of_label_algo_calls[sub_graph_id] += 1
            #list_of_fallbacks[sub_graph_id] += 1

        #end
        if (list_of_sub_graphs[sub_graph_id].static & (length(path_list) < 1)) 
            list_of_sub_graphs[sub_graph_id].static = false
            #list_of_sub_graphs[sub_graph_id].no_paths_found = false
        end
        #list_of_Edges_ID_lists[sub_graph_id] = edges_id_list 
        list_of_path_lists[sub_graph_id] = path_list
    end
end

function semi_stoc_thread_call_labeling_algo_oall!_partial(list_of_path_lists::Vector{Vector{Path}},list_of_fallbacks::Vector{Int},list_of_label_algo_calls::Vector{Int},  list_of_sub_graphs::Vector{GraphSub}, thread_indexes::Vector{Int}, G::Graph, threshold::Float64,choice_function::Function, max_paths_in_pricing = 1000,  model = nothing, model_dyn = nothing)
    for sub_graph_id in thread_indexes
        #println("sub_graph ", sub_graph_id, " by thread ", Threads.threadid())
        no_paths, max_dual = list_of_sub_graphs[sub_graph_id].update_dual(G)
        if no_paths
            continue
        end        #edges_id_list = [Vector{Int}() for i in 1:G.number_of_edges]
        #if list_of_sub_graphs[sub_graph_id].no_paths_found 
        #    model_dyn = nothing
        #end


        if isnothing(model_dyn)
            
        elseif !list_of_sub_graphs[sub_graph_id].static
            list_of_sub_graphs[sub_graph_id].predict(model_dyn)
#           list_of_sub_graphs[sub_graph_id].no_paths_found = false
        end
        path_list = Vector{Path}()
        if list_of_sub_graphs[sub_graph_id].static & isnothing(model)
            semi_stoc_labeling_algorithm_opt!_partial(G, path_list, list_of_sub_graphs[sub_graph_id], 0.0, choice_function, max_dual, max_paths_in_pricing)
            list_of_label_algo_calls[sub_graph_id] += 1
        elseif list_of_sub_graphs[sub_graph_id].static & !isnothing(model)
            semi_stoc_labeling_algorithm_opt!_partial(G, path_list, list_of_sub_graphs[sub_graph_id], 0.5, choice_function, max_dual, max_paths_in_pricing)
            list_of_label_algo_calls[sub_graph_id] += 1
        elseif !list_of_sub_graphs[sub_graph_id].static & isnothing(model_dyn)
            semi_stoc_labeling_algorithm_opt!_partial(G, path_list, list_of_sub_graphs[sub_graph_id], 0.0, choice_function, max_dual, max_paths_in_pricing)
            list_of_label_algo_calls[sub_graph_id] += 1
        else
            semi_stoc_labeling_algorithm_opt!_partial(G, path_list, list_of_sub_graphs[sub_graph_id], list_of_sub_graphs[sub_graph_id].threshold, choice_function, max_dual, max_paths_in_pricing)
            list_of_label_algo_calls[sub_graph_id] += 1
        end
        if !list_of_sub_graphs[sub_graph_id].static & !isnothing(model_dyn) & (length(path_list) < 1)
            #list_of_sub_graphs[sub_graph_id].threshold -= 0.1
            list_of_sub_graphs[sub_graph_id].no_paths_found = true
        end

        #if (length(path_list) < 1) & !isnothing(model) & !list_of_sub_graphs[sub_graph_id].no_paths_found
            #list_of_sub_graphs[sub_graph_id].no_paths_found = true
            #semi_stoc_labeling_algorithm_opt!(G, path_list, list_of_sub_graphs[sub_graph_id], 0.0, choice_function, max_dual)
            #list_of_label_algo_calls[sub_graph_id] += 1
            #list_of_fallbacks[sub_graph_id] += 1

        #end
        if (list_of_sub_graphs[sub_graph_id].static) & (length(path_list) < 1) 
            list_of_sub_graphs[sub_graph_id].static = false
            #list_of_sub_graphs[sub_graph_id].no_paths_found = false
        end
        #list_of_Edges_ID_lists[sub_graph_id] = edges_id_list 
        list_of_path_lists[sub_graph_id] = path_list
    end
end

function solve_sub_problems_oall_partial_pricing(list_of_sub_graphs::Vector{GraphSub}, G::Graph,  threshold, choice_function, num_sub, max_paths, process_order, max_paths_in_pricing = 1000, nr_threads = Threads.nthreads(), model = nothing, model_dyn = nothing)
    countr = Threads.Atomic{Int}(1)
    path_countr = Threads.Atomic{Int}(0)
    nr_sub_graphs = length(list_of_sub_graphs)
    list_of_path_lists = [Vector{Path}() for i in 1:nr_sub_graphs]
    list_of_fallbacks = [0 for i in 1:nr_sub_graphs]
    list_of_label_algo_calls = [0 for i in 1:nr_sub_graphs]

    Threads.@threads for i in 1:nr_threads
        while true
            ind = Threads.atomic_add!(countr, 1)
            lel = Threads.atomic_add!(path_countr, 1)
            if ((ind +1) > length(list_of_sub_graphs)) || ((lel+1) > max_paths)
                break
            end
            sub_ind = (((ind+1) + num_sub) % length(list_of_sub_graphs))+1 
            #println(sub_ind)
            semi_stoc_thread_call_labeling_algo_oall!_partial(list_of_path_lists,list_of_fallbacks,list_of_label_algo_calls,  list_of_sub_graphs, [process_order[sub_ind]], G, threshold,  choice_function,max_paths_in_pricing, model, model_dyn)
        end
    end
    return list_of_path_lists, sum(list_of_fallbacks), sum(list_of_label_algo_calls), (((countr[] + num_sub) % length(list_of_sub_graphs))+1)
end



function solve_sub_problems_oall(list_of_sub_graphs::Vector{GraphSub}, G::Graph,  threshold, choice_function, nr_threads = Threads.nthreads(), model = nothing, model_dyn = nothing)
    countr = Threads.Atomic{Int}(1)
    nr_sub_graphs = length(list_of_sub_graphs)
    list_of_path_lists = [Vector{Path}() for i in 1:nr_sub_graphs]
    list_of_fallbacks = [0 for i in 1:nr_sub_graphs]
    list_of_label_algo_calls = [0 for i in 1:nr_sub_graphs]

    Threads.@threads for i in 1:nr_threads
        while true
            ind = Threads.atomic_add!(countr, 1)
            if ind > length(list_of_sub_graphs)
                break
            end
            semi_stoc_thread_call_labeling_algo_oall!(list_of_path_lists,list_of_fallbacks,list_of_label_algo_calls,  list_of_sub_graphs, [ind], G, threshold, choice_function, model, model_dyn)
        end
    end
    return list_of_path_lists, sum(list_of_fallbacks), sum(list_of_label_algo_calls)
end

