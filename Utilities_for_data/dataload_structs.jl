using DataFrames
using CSV

function struct_to_dataframe(list_of_obs)
    fields = fieldnames(typeof(list_of_obs[1]))
    still_vector = [getfield(list_of_obs[1],field) for field in fields]

    for s in 2:length(list_of_obs)
        for i in 1:size(still_vector)[1]
            append!(still_vector[i], getfield(list_of_obs[s], fields[i]))
        end
    end

    return DataFrame(hcat(still_vector...), collect(fields))
end

function obs_to_floats(obs)
    to_remove = ["in_path", "in_opt", "instance_name","edge_id","iteration","sub_graph_id","lambda_ij","arcs_lambda_i_sub_min","arcs_lambda_i_sub_max", "arcs_lambda_i_sub_mean","arcs_lambda_j_sub_min","arcs_lambda_j_sub_max","arcs_lambda_j_sub_mean","pi_ij","times_chosen_for_sub","meanpj","max_pj","min_pj","sink_edge","load_edge","forfeit_edge", "dist_less", "num_dest"]
    to_keep =  ["cost_of_edge", "time_of_edge", "cap_of_edge", "arc_leaving_i_sub", "arc_leaving_j_sub", "arc_entering_i_sub", "arc_entering_j_sub", "arcs_cost_i_sub_min", "arcs_cost_i_sub_mean", "arcs_time_i_sub_min", "arcs_time_i_sub_mean", "arcs_cap_i_sub_min", "arcs_cap_i_sub_mean", "arcs_ent_cost_i_sub_min", "arcs_ent_cost_i_sub_mean", "arcs_ent_time_i_sub_min", "arcs_ent_time_i_sub_mean", "arcs_ent_cap_i_sub_min", "arcs_ent_cap_i_sub_mean", "arcs_cost_j_sub_min", "arcs_cost_j_sub_mean", "arcs_time_j_sub_min", "arcs_time_j_sub_mean", "arcs_cap_j_sub_min", "arcs_cap_j_sub_mean", "arcs_ent_cost_j_sub_min", "arcs_ent_cost_j_sub_mean", "arcs_ent_time_j_sub_min", "arcs_ent_time_j_sub_mean", "arcs_ent_cap_j_sub_min", "arcs_ent_cap_j_sub_mean", "voy_edge", "trans_edge"]


    fields = fieldnames(typeof(obs))

    still_vector = [getfield(obs,field) for field in fields if ((!contains(String(field),"base")) & (String(field) ∉ to_remove)&(String(field) ∈ to_keep))]
    
    return Float32.(transpose(hcat(still_vector...)))
end

function obs_to_floats_dynamic(obs)
    to_remove = ["in_path", "in_opt", "instance_name","edge_id","iteration","sub_graph_id","times_chosen_for_sub", "sink_edge","load_edge","forfeit_edge","dist_less", "num_dest"]
    
    to_keep = ["cost_of_edge", "time_of_edge", "cap_of_edge", "arc_leaving_i_sub", "arc_leaving_j_sub", "arc_entering_i_sub", "arc_entering_j_sub", "arcs_cost_i_sub_min", "arcs_cost_i_sub_mean", "arcs_time_i_sub_min", "arcs_time_i_sub_mean", "arcs_cap_i_sub_min", "arcs_cap_i_sub_mean", "arcs_ent_cost_i_sub_min", "arcs_ent_cost_i_sub_mean", "arcs_ent_time_i_sub_min", "arcs_ent_time_i_sub_mean", "arcs_ent_cap_i_sub_min", "arcs_ent_cap_i_sub_mean", "arcs_cost_j_sub_min", "arcs_cost_j_sub_mean", "arcs_time_j_sub_min", "arcs_time_j_sub_mean", "arcs_cap_j_sub_min", "arcs_cap_j_sub_mean", "arcs_ent_cost_j_sub_min", "arcs_ent_cost_j_sub_mean", "arcs_ent_time_j_sub_min", "arcs_ent_time_j_sub_mean", "arcs_ent_cap_j_sub_min", "arcs_ent_cap_j_sub_mean", "voy_edge", "trans_edge", "lambda_ij", "arcs_lambda_i_sub_min", "arcs_lambda_i_sub_mean", "arcs_lambda_j_sub_min", "arcs_lambda_j_sub_mean", "arcs_ent_lambda_i_sub_min", "arcs_ent_lambda_i_sub_mean", "arcs_ent_lambda_j_sub_min", "arcs_ent_lambda_j_sub_mean"] 

    fields = fieldnames(typeof(obs))

    still_vector = [getfield(obs,field) for field in fields if ((!contains(String(field),"base")) & (String(field) ∉ to_remove) &(String(field) ∈ to_keep))]
    
    return Float32.(transpose(hcat(still_vector...)))
end

