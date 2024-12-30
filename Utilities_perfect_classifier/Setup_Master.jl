using JuMP, GLPK, Gurobi
using Base.Threads

function init_forfeit_paths(env::Env, G::Graph)
    path_list_master = Vector{Path}()
    voy_edges_cap_list = [G.edges_ordered[i].capacity for i in G.voy_edge_start_id:G.voy_edge_end_id]
    for (ports, list) in G.demand_od_path_dict
        new_path = Path(0, [0],[1000], Vector{Int}(), 1000, ports[2], ports[1], list.id)   
        push!(path_list_master, new_path)
    end
    return path_list_master, voy_edges_cap_list
end

function setup_master(optimizer::DataType, path_list_master::Vector{Path}, voy_edges_cap_list::Vector{Float64}, demand_list::Vector{Demand})
    master = JuMP.Model(optimizer)
    @variable(master, x[1:length(path_list_master)]>=0)
    @objective(master,Min, sum(x[d]*path_list_master[d].cost for d in 1:length(x)))
    @constraint(master, edges_cons[e = 1:length(voy_edges_cap_list)], 0 <= voy_edges_cap_list[e])
    @constraint(master, demand_cons[d = 1:length(demand_list)], 0 == demand_list[d].ffe_per_week)
    for i in 1:length(x)
        path = path_list_master[i]
        set_normalized_coefficient(demand_cons[path.demand_id], x[i], 1)
    end
    return master, x, edges_cons, demand_cons, 0
end



function setup_master_2(path_list_master::Vector{Path}, voy_edges_cap_list::Vector{Float64}, demand_list::Vector{Demand})
    num_paths = length(demand_list)
    master = Model(GLPK.Optimizer)
    @variable(master, x[1:40000]>=0)
    @objective(master,Min, sum(x[d]*path_list_master[d].cost for d in 1:length(demand_list)))
    @constraint(master, edges_cons[e = 1:length(voy_edges_cap_list)], 0 <= voy_edges_cap_list[e])
    @constraint(master, demand_cons[d = 1:length(demand_list)], 0 == demand_list[d].ffe_per_week)
    for i in 1:length(demand_list)
        path = path_list_master[i]
        set_normalized_coefficient(demand_cons[path.demand_id], x[i], 1)
    end
    return master, x, edges_cons, demand_cons, num_paths
end


function add_paths_to_master_inplace(path_list::Vector{Path}, demand_list::Vector{Demand}, master::JuMP.Model, edges_cons, demand_cons, x, num_paths)

    for (i, path) in enumerate(path_list)
        JuMP.set_objective_coefficient(master, x[i+num_paths], path.cost - demand_list[path.demand_id].revenue)
        for voy_id in path.voy_edges
            set_normalized_coefficient(edges_cons[voy_id], x[i+num_paths], 1)
        end
        set_normalized_coefficient(demand_cons[path.demand_id], x[i+num_paths], 1)

    end
    num_paths += length(path_list)
    return num_paths
end


function add_paths_to_master(path_list::Vector{Path}, demand_list::Vector{Demand}, master::JuMP.Model, edges_cons, demand_cons, x, dum_n)
    len_of_x = length(x)
    
    
    for (i, path) in enumerate(path_list)
        new_var = @variable(master, base_name = "x[$(len_of_x+i)]", lower_bound = 0)
        push!(x, new_var)
        JuMP.set_objective_coefficient(master, new_var, path.cost - demand_list[path.demand_id].revenue)
        for voy_id in path.voy_edges
            set_normalized_coefficient(edges_cons[voy_id], new_var, 1)
        end
        set_normalized_coefficient(demand_cons[path.demand_id], new_var, 1)

    end
    return 0
end
