include("Struct_sub_graph.jl")
include("Label_algo_optimized.jl")
include("ReadLogs.jl")
include("Structs.jl")
include("Parallel_Solve.jl")
include("Setup_Master.jl")
#include("import_model.jl")
include("dataload_structs.jl")
include("Choice_models.jl")
using BSON: @save,@load
using Flux, CUDA, Statistics, ProgressMeter,Distributions

function confidence_interval(data::Vector{T}, confidence_level=0.95) where T
    n = length(data)
    mean_value = mean(data)
    std_err = std(data) / sqrt(n)
    
    # Get the t-critical value based on the confidence level
    t_critical = quantile(TDist(n - 1), 1 - (1 - confidence_level) / 2)
    
    margin_of_error = t_critical * std_err
    
    lower_bound = mean_value - margin_of_error
    upper_bound = mean_value + margin_of_error
    
    return lower_bound, upper_bound
end
mutable struct Experiment
    instance_name :: String
    choice_model :: String
    optimizer :: String
    threshold :: Float64
    static_model_name :: String
    dynamic_model_name :: String
    objective_value :: Vector{Float64}
    time_spent_in_pricing :: Vector{Float64}
    time_spent_adding_paths :: Vector{Float64}
    time_spent_in_master :: Vector{Float64}
    time_spent_total :: Vector{Float64}
    number_of_label_algo_calls :: Vector{Int}
    number_of_fall_backs :: Vector{Int}
    number_of_paths :: Vector{Int}
    number_of_iterations :: Vector{Int}
    vector_of_it_of_time_spent_in_pricing :: Vector{Vector{Float64}}
    vector_of_it_of_time_spent_adding_paths :: Vector{Vector{Float64}}
    vector_of_it_of_time_spent_in_master :: Vector{Vector{Float64}}
    vector_of_it_of_label_algo_calls :: Vector{Vector{Int}}
    vector_of_it_of_fall_backs :: Vector{Vector{Int}}
    vector_of_it_of_paths :: Vector{Vector{Int}}
end
function Base.getproperty(this::Experiment, s::Symbol)
    if s ==:format_2_row
        function(filename::String,append = true)
            vect_field_dict = Dict()
            vector_fields =[:vector_of_it_of_time_spent_in_pricing
                            :vector_of_it_of_time_spent_adding_paths
                            :vector_of_it_of_time_spent_in_master
                            :vector_of_it_of_label_algo_calls
                            :vector_of_it_of_fall_backs
                            :vector_of_it_of_paths
                            ]
            for field in vector_fields
                data = getfield(this,field)
                
                max_length = maximum(length.(data))
                padded_data = [vcat(vec, fill(missing, max_length - length(vec))) for vec in data]
                data_matrix = reduce(hcat, padded_data)
                vect_field_dict[field] = Tuple([mean(skipmissing(data_matrix[i,:])) for i in 1:size(data_matrix, 1)])
            end

            new_row = DataFrame(:instance_name => this.instance_name,
                                :optimizer => this.optimizer,
                                :choice_model => this.choice_model,
                                :threshold => this.threshold,
                                :static_model_name => this.static_model_name,
                                :dynamic_model_name => this.dynamic_model_name,
                                :objective_value => mean(this.objective_value),

                                :time_spent_in_pricing_mean => mean(this.time_spent_in_pricing), 
                                :time_spent_adding_paths_mean => mean(this.time_spent_adding_paths),  
                                :time_spent_in_master_mean => mean(this.time_spent_in_master),  
                                :time_spent_total_mean => mean(this.time_spent_total),  
                                :number_of_label_algo_calls_mean => mean(this.number_of_label_algo_calls), 
                                :number_of_fall_backs_mean => mean(this.number_of_fall_backs),  
                                :number_of_paths_mean => mean(this.number_of_paths),  
                                :number_of_iterations_mean => mean(this.number_of_iterations),

                                :time_spent_in_pricing_CI => confidence_interval(this.time_spent_in_pricing), 
                                :time_spent_adding_paths_CI => confidence_interval(this.time_spent_adding_paths),  
                                :time_spent_in_master_CI => confidence_interval(this.time_spent_in_master),  
                                :time_spent_total_CI => confidence_interval(this.time_spent_total),  
                                :number_of_label_algo_calls_CI => confidence_interval(this.number_of_label_algo_calls), 
                                :number_of_fall_backs_CI => confidence_interval(this.number_of_fall_backs),  
                                :number_of_paths_CI => confidence_interval(this.number_of_paths),  
                                :number_of_iterations_CI => confidence_interval(this.number_of_iterations),
                                
                                :vector_of_it_of_time_spent_in_pricing => vect_field_dict[:vector_of_it_of_time_spent_in_pricing], 
                                :vector_of_it_of_time_spent_adding_paths => vect_field_dict[:vector_of_it_of_time_spent_adding_paths],
                                :vector_of_it_of_time_spent_in_master => vect_field_dict[:vector_of_it_of_time_spent_in_master],
                                :vector_of_it_of_label_algo_calls => vect_field_dict[:vector_of_it_of_label_algo_calls],
                                :vector_of_it_of_fall_backs => vect_field_dict[:vector_of_it_of_fall_backs],
                                :vector_of_it_of_paths => vect_field_dict[:vector_of_it_of_paths],
            )                   

            

            if append

                if filesize(filename) == 0
                    CSV.write(filename,new_row)
                else
                    open(filename, "a") do io
                        CSV.write(io, new_row, append=true)
                    end
                end
            end
            return new_row
        end
    else getfield(this, s)
        
    end
end
function main_o_all_experiment_call(experiment:: Experiment, optimizer:: DataType, demand_file::String, instance_file::String,static::Bool, model = nothing, model_dyn = nothing, threshold = 0.5, choice_model = no_choice, partial_pricing = false)

    time_spent_in_pricing = 0.0
    time_spent_adding_paths = 0.0
    time_spent_in_master = 0.0
    time_spent_total = 0.0
    number_of_label_algo_calls = 0
    number_of_fall_backs = 0
    number_of_paths = 0
    number_of_iterations = 0
    vector_of_it_of_time_spent_in_pricing = Vector{Float64}()
    vector_of_it_of_time_spent_adding_paths = Vector{Float64}()
    vector_of_it_of_time_spent_in_master = Vector{Float64}()
    vector_of_it_of_label_algo_calls = Vector{Int}()
    vector_of_it_of_fall_backs = Vector{Int}()
    vector_of_it_of_paths = Vector{Int}()

    env = create_env(10000.0,2.0,1.0, demand_file, instance_file, 100.0, 1.0)
    G = Graph(Dict{String, Vector{Node}}(), Dict{String, Node}(), Dict{Tuple{Int64, Int64}, Edge}(),Vector{Edge}(), Vector{Node}(),Vector{Vector{Node}}(), 0, Vector{Int64}(), 0,0, Vector{Node}(), 0, Dict{Tuple{String,String}, Vector{Int}}(), Dict{Tuple{String,String}, Float64}(), 0, 0)
    G.build_star_graph(env)

    list_of_sub_graphs = create_sub_problems_o_all(env, G)
    for sub in list_of_sub_graphs
        sub.threshold = threshold
    end
    process_order = sortperm([list_of_sub_graphs[i].total_demand for i in 1:length(list_of_sub_graphs)], rev = true)

    path_list_master, voy_edges_cap_list =  init_forfeit_paths(env, G)

    total_time = time()

    master, x, edges_cons, demand_cons, num_paths = setup_master(optimizer, path_list_master, voy_edges_cap_list, env.demand_list)
    number_of_paths = num_paths

    time_spent_in_ma = time()
    optimize!(master)
    time_spent_in_master += time() - time_spent_in_ma
    push!(vector_of_it_of_time_spent_in_master, time() - time_spent_in_ma)
    pis = dual.(demand_cons)
    lambdas = dual.(edges_cons)
    G.update_lambda(lambdas)
    G.update_pi(pis)

    if !partial_pricing
        time_sub = time()
        list_of_path_lists, fallbacks, label_algo_calls  = solve_sub_problems_oall(list_of_sub_graphs, G,  threshold, choice_model, 16, model, model_dyn)
        elapsed_time = time() - time_sub
        time_spent_in_pricing += elapsed_time
    else 

        time_sub = time()
        list_of_path_lists, fallbacks, label_algo_calls, num_sub  = solve_sub_problems_oall_partial_pricing(list_of_sub_graphs, G,  threshold, choice_model, 1 ,2000, process_order, 400, 16, model, model_dyn)
        elapsed_time = time() - time_sub
        time_spent_in_pricing += elapsed_time
    end
    push!(vector_of_it_of_time_spent_in_pricing, elapsed_time)
    push!(vector_of_it_of_fall_backs, fallbacks)
    push!(vector_of_it_of_label_algo_calls, label_algo_calls)
    number_of_label_algo_calls += label_algo_calls
    number_of_fall_backs += fallbacks

    path_list = syncronize_all(list_of_path_lists)
    number_of_iterations = 1
    fallback = false

    while (length(path_list)>0) || fallback
        if fallback
            fallback = false
        end
        number_of_paths += length(path_list)
        push!(vector_of_it_of_paths, length(path_list))

        t1 = time()
        num_paths = add_paths_to_master(path_list, env.demand_list, master, edges_cons, demand_cons, x, num_paths)
        elapsed_time = time() - t1
        time_spent_adding_paths += elapsed_time
        push!(vector_of_it_of_time_spent_adding_paths, elapsed_time) 

        t1 = time()
        optimize!(master)
        elapsed_time = time() - t1
        
        time_spent_in_master += elapsed_time
        push!(vector_of_it_of_time_spent_in_master, elapsed_time) 


        pis = dual.(demand_cons)
        lambdas = dual.(edges_cons)
        G.update_lambda(lambdas)
        G.update_pi(pis)

        if !partial_pricing    
            time_sub = time()

            list_of_path_lists, fallbacks, label_algo_calls = solve_sub_problems_oall(list_of_sub_graphs, G, threshold, choice_model, 16, model, model_dyn)
            elapsed_time = time() - time_sub
            time_spent_in_pricing += elapsed_time
        else 

            time_sub = time()
            list_of_path_lists, fallbacks, label_algo_calls, num_sub  = solve_sub_problems_oall_partial_pricing(list_of_sub_graphs, G,  threshold, choice_model, num_sub, 1000, process_order, 1000, 16, model, model_dyn)
            elapsed_time = time() - time_sub
            time_spent_in_pricing += elapsed_time
        end
        
        push!(vector_of_it_of_time_spent_in_pricing, elapsed_time)
        push!(vector_of_it_of_fall_backs, fallbacks)
        push!(vector_of_it_of_label_algo_calls, label_algo_calls)
        number_of_label_algo_calls += label_algo_calls
        number_of_fall_backs += fallbacks
        path_list = syncronize_all(list_of_path_lists)
        if length(path_list) < 200 & !isnothing(model_dyn)
            println("fallback")
            for sub in list_of_sub_graphs
                sub.threshold = 0.0
            end
            model_dyn = nothing
            #model = nothing
            #model_dyn = nothing
            #fallback = false
            #threshold = 0.0
        end
        number_of_iterations +=1
    end
    time_spent_total = time() - total_time

    push!(experiment.objective_value, objective_value(master))
    push!(experiment.time_spent_in_pricing, time_spent_in_pricing) 
    push!(experiment.time_spent_adding_paths, time_spent_adding_paths) 
    push!(experiment.time_spent_in_master, time_spent_in_master)  
    push!(experiment.time_spent_total, time_spent_total)  
    push!(experiment.number_of_label_algo_calls, number_of_label_algo_calls) 
    push!(experiment.number_of_fall_backs, number_of_fall_backs)  
    push!(experiment.number_of_paths, number_of_paths)  
    push!(experiment.number_of_iterations, number_of_iterations)  
    push!(experiment.vector_of_it_of_time_spent_in_pricing, vector_of_it_of_time_spent_in_pricing)  
    push!(experiment.vector_of_it_of_time_spent_adding_paths, vector_of_it_of_time_spent_adding_paths) 
    push!(experiment.vector_of_it_of_time_spent_in_master, vector_of_it_of_time_spent_in_master) 
    push!(experiment.vector_of_it_of_label_algo_calls, vector_of_it_of_label_algo_calls)  
    push!(experiment.vector_of_it_of_fall_backs, vector_of_it_of_fall_backs)  
    push!(experiment.vector_of_it_of_paths, vector_of_it_of_paths)  
    return 
end

function experiment_loop(trials :: Int, optimizer_name:: String, demand_file::String, instance_file::String,static::Bool, model_name::String , model_dyn_name::String, threshold = 0.5, choice_model = no_choice, partial_pricing = false)
    experiment = Experiment(instance_file, String(Symbol(choice_model)), optimizer_name,threshold, model_name, model_dyn_name,Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), 
                                        Vector{Float64}(), Vector{Float64}(), Vector{Int}(),  Vector{Int}(),  Vector{Int}(),  Vector{Int}(), 
                                        Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), Vector{Vector{Float64}}(),
                                        Vector{Vector{Int}}(), Vector{Vector{Int}}(), Vector{Vector{Int}}())
    
    if model_dyn_name == "nothing"
        model_dyn = nothing
        model = nothing
    else
        @load "NewModels/"*model_dyn_name model
        model_dyn = deepcopy(model)
        if model_name == "nothing"
            model = nothing 
        else
            @load "NewModels/"*model_name model
        end
    end
    if optimizer_name == "Gurobi"
        for i = 1:trials
            main_o_all_experiment_call(experiment, Gurobi.Optimizer, demand_file, instance_file, false, model, model_dyn, threshold, choice_model, partial_pricing)
        end
    else
        for i = 1:trials

            main_o_all_experiment_call(experiment, GLPK.Optimizer, demand_file, instance_file, false, model, model_dyn, threshold, choice_model, partial_pricing)
    
        end
    end
    return experiment
end

experiment_no_partial = experiment_loop(5, "Gurobi", "Demand_WorldLarge", "Instance-WL.txt", false,"nothing", "nothing", 0.0)
experiment_no_partial.format_2_row("partial_test_no.csv", true)

experiment_partial = experiment_loop(5, "Gurobi", "Demand_WorldLarge", "Instance-WL.txt", false,"nothing", "nothing", 0.0, no_choice, true)
experiment_partial.format_2_row("partial_test_yes.csv", true)







function main()
    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    files = Dict("Demand_EuropeAsia"=> ["EuropeAsia_pid_824_4_best_high", "EuropeAsia_pid_26674_12", "EuropeAsia_pid_26999_11_low_best"])
    choice_models = [choose_10p_less, choose_5p_less, test_choice, no_choice]
    demands = ["Demand_EuropeAsia"]
    models = Dict(
        "Demand_EuropeAsia" => [("EuropeAsia_pid_824_4_best_high_static_t100.csv.bson","EuropeAsia_pid_824_4_best_high_dynamic_t100.csv.bson"),("EuropeAsia_pid_26674_12_static_t100.csv.bson","EuropeAsia_pid_26674_12_dynamic_t100.csv.bson"), ( "EuropeAsia_pid_26999_11_low_best_static_t100.csv.bson", "EuropeAsia_pid_26999_11_low_best_dynamic_t100.csv.bson")])

    for demand in demands
        for (i,file) in enumerate(files[demand]) 
            experiment00 = experiment_loop(10, "Gurobi", demand, file, false,"nothing", "nothing", 0.0, no_choice, true)
            experiment00.format_2_row("partial_no_model.csv", true)
            for choice_model in choice_models
                for threshold in thresholds
                    experiment05 = experiment_loop(10, "Gurobi", demand, file, false, "nothing", models[demand][i][2], threshold, choice_model, true)
                    experiment05.format_2_row("partial_with_model_no_fallback.csv", true)
                end
            end     
        end
    end
end

main()