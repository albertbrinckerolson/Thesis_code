include("ReadLogs.jl")
include("Structs.jl")

mutable struct EdgeSub
    ij :: Tuple{Int64, Int64}
    id :: Int
    cost :: Float64
    time :: Float64
    dual :: Float64
    prob :: Float64
    edges_in_base:: Vector{Int}
    obs_id :: Int64
    type::String
end



mutable struct NodeSub
    id :: Int64
    port :: String
    outgoing_edges :: Vector{EdgeSub}
    ingoing_edges :: Vector{EdgeSub}
    type::String
end

mutable struct Observations
    
    #Prediction
    in_path :: Vector{Int}
    in_opt :: Vector{Int}

    #Identification_features
    instance_name :: Vector{String}
    edge_id :: Vector{Int}
    iteration :: Vector{Int}
    sub_graph_id :: Vector{Int}

    #Base_graph_features
        #i-node
    arc_leaving_i_base :: Vector{Float64}
    arc_leaving_j_base :: Vector{Float64}
    arcs_cost_i_base_min :: Vector{Float64}
    arcs_cost_i_base_max :: Vector{Float64}
    arcs_cost_i_base_mean :: Vector{Float64}
    arcs_time_i_base_min :: Vector{Float64}
    arcs_time_i_base_max :: Vector{Float64}
    arcs_time_i_base_mean :: Vector{Float64}

        #j-node 
    arcs_cost_j_base_min :: Vector{Float64}
    arcs_cost_j_base_max :: Vector{Float64}
    arcs_cost_j_base_mean :: Vector{Float64}
    arcs_time_j_base_min :: Vector{Float64}
    arcs_time_j_base_max :: Vector{Float64}
    arcs_time_j_base_mean :: Vector{Float64}
    dist_less:: Vector{Float64}

    #Sub_graph_features
    cost_of_edge :: Vector{Float64}
    time_of_edge :: Vector{Float64}
    cap_of_edge :: Vector{Float64}
    arc_leaving_i_sub :: Vector{Float64}
    arc_leaving_j_sub :: Vector{Float64}
    arc_entering_i_sub :: Vector{Float64}
    arc_entering_j_sub :: Vector{Float64}

    dest_at_i :: Vector{Float64}
    dest_at_j :: Vector{Float64}

        #i-node
    arcs_cost_i_sub_min :: Vector{Float64}
    arcs_cost_i_sub_max :: Vector{Float64}
    arcs_cost_i_sub_mean :: Vector{Float64}
    arcs_time_i_sub_min :: Vector{Float64}
    arcs_time_i_sub_max :: Vector{Float64}
    arcs_time_i_sub_mean :: Vector{Float64}
    arcs_cap_i_sub_min :: Vector{Float64}
    arcs_cap_i_sub_max :: Vector{Float64}
    arcs_cap_i_sub_mean :: Vector{Float64}

    arcs_ent_cost_i_sub_min :: Vector{Float64}
    arcs_ent_cost_i_sub_max :: Vector{Float64}
    arcs_ent_cost_i_sub_mean :: Vector{Float64}
    arcs_ent_time_i_sub_min :: Vector{Float64}
    arcs_ent_time_i_sub_max :: Vector{Float64}
    arcs_ent_time_i_sub_mean :: Vector{Float64}
    arcs_ent_cap_i_sub_min :: Vector{Float64}
    arcs_ent_cap_i_sub_max :: Vector{Float64}
    arcs_ent_cap_i_sub_mean :: Vector{Float64}
    

        #j-node 
    arcs_cost_j_sub_min :: Vector{Float64}
    arcs_cost_j_sub_max :: Vector{Float64}
    arcs_cost_j_sub_mean :: Vector{Float64}
    arcs_time_j_sub_min :: Vector{Float64}
    arcs_time_j_sub_max :: Vector{Float64}
    arcs_time_j_sub_mean :: Vector{Float64}
    arcs_cap_j_sub_min :: Vector{Float64}
    arcs_cap_j_sub_max :: Vector{Float64}
    arcs_cap_j_sub_mean :: Vector{Float64}
    
    arcs_ent_cost_j_sub_min :: Vector{Float64}
    arcs_ent_cost_j_sub_max :: Vector{Float64}
    arcs_ent_cost_j_sub_mean :: Vector{Float64}
    arcs_ent_time_j_sub_min :: Vector{Float64}
    arcs_ent_time_j_sub_max :: Vector{Float64}
    arcs_ent_time_j_sub_mean :: Vector{Float64}
    arcs_ent_cap_j_sub_min :: Vector{Float64}
    arcs_ent_cap_j_sub_max :: Vector{Float64}
    arcs_ent_cap_j_sub_mean :: Vector{Float64}

    num_dest :: Vector{Float64}

    voy_edge :: Vector{Float64}
    trans_edge :: Vector{Float64}
    load_edge :: Vector{Float64}
    forfeit_edge :: Vector{Float64}
    sink_edge :: Vector{Float64}
    

    #Dynamic features
    lambda_ij :: Vector{Float64}

        #i-node
    arcs_lambda_i_sub_min :: Vector{Float64}
    arcs_lambda_i_sub_max :: Vector{Float64}
    arcs_lambda_i_sub_mean :: Vector{Float64}
        #j-node
    arcs_lambda_j_sub_min :: Vector{Float64}
    arcs_lambda_j_sub_max :: Vector{Float64}
    arcs_lambda_j_sub_mean :: Vector{Float64}

    arcs_ent_lambda_i_sub_min :: Vector{Float64}
    arcs_ent_lambda_i_sub_mean :: Vector{Float64}
    arcs_ent_lambda_j_sub_min :: Vector{Float64}
    arcs_ent_lambda_j_sub_mean :: Vector{Float64}

    pi_ij :: Vector{Float64}
    times_chosen_for_sub :: Vector{Float64}
    meanpj :: Vector{Float64}
    max_pj :: Vector{Float64}
    min_pj :: Vector{Float64}


end



mutable struct GraphSub
    id :: Int
    edges_list :: Dict{Tuple{Int64, Int64}, EdgeSub}
    edges_ordered :: Vector{EdgeSub}
    node_dict :: Dict{Int, NodeSub}
    source_port :: String
    source_base_id :: Int
    destinations :: Set{String}
    number_of_nodes :: Int64
    max_time :: Float64
    sink_id :: Int
    data_frame :: Observations
    no_paths_found :: Bool
    static :: Bool
    total_demand :: Int
    threshold :: Float64
    times_none_found :: Int
    max_rev :: Float64
end

function min_max_normalize!(x::Vector{Float64}, field::String, Gs::GraphSub)
    min = minimum(x)
    if occursin("time", field)
        max = Gs.max_time
    elseif occursin("cost", field)
        max = Gs.max_rev
    else
        max = maximum(x)
    end
    for i in 1:length(x)
        if max-min != 0
            x[i] = (x[i]-min)/(max-min) 
        else 
            x[i] = 0
        end
    end
end


function Base.getproperty(this::Observations, s::Symbol)
    if s== :normalize
        function(Gs::GraphSub)
            fields = fieldnames(Observations)
            for field in fields[7:end]
                min_max_normalize!(getfield(this, field), String(field), Gs)
            end
        end
    elseif s==:normalize_dyn
        function(Gs::GraphSub)
            fields = fieldnames(Observations)
            for field in fields[end-16:end]
                min_max_normalize!(getfield(this, field), String(field), Gs)
            end
        end
    else
        getfield(this, s)
    end
end

function Base.getproperty(this::GraphSub, s::Symbol)
    if s == :add_node 
        function(port::String, type::String, id::Int)
            this.number_of_nodes += 1
            new_node = NodeSub(id, port, Vector{EdgeSub}(),Vector{EdgeSub}(), type)
            this.node_dict[id] = new_node
            return new_node
        end
        
    elseif  s == :add_edge
        function(node1id::Int64, node2id::Int64, cost::Float64, time::Float64, edges_in_base::Vector{Int}, type::String)
            id = length(this.edges_ordered)+1
            new_Edge = EdgeSub((node1id, node2id),id, cost, time, 0,  0, edges_in_base, 0, type)
            push!(this.edges_ordered, new_Edge)
            this.edges_list[new_Edge.ij] = new_Edge
            push!(this.node_dict[node1id].outgoing_edges, new_Edge)
            push!(this.node_dict[node2id].ingoing_edges, new_Edge)

        end

    elseif s==:load_source_and_demand
        function(G::Graph)
            source = this.add_node(this.source_port, "source_node", this.source_base_id)
            #add_loads from source
            for load_node in G.call_node_dict[source.port]
                cost, time = G.get_edge_cost_and_time(source.id, load_node.id)
                call_node = this.add_node(load_node.port, "call_node", load_node.id)
                this.add_edge(source.id, load_node.id, cost, time, [G.edges_list[(source.id, load_node.id)].id], "load_edge")
            end

            sink = this.add_node("SINK", "sink_node", this.sink_id) 

            for (key, node) in G.port_dict
                if key in this.destinations
                    dest_node = this.add_node(node.port, "destination_node", node.id)
                    for load_node in G.call_node_dict[node.port]
                        cost, time = G.get_edge_cost_and_time(load_node.id, node.id)
                        call_node = this.add_node(node.port, "call_node", load_node.id)
                        this.add_edge(call_node.id, node.id, cost, time, [G.edges_list[(call_node.id, node.id)].id], "load_edge")
                    end
                    #new_forfeit = G.edges_list[source.id, node.id]
                    #this.add_edge(source.id, node.id, new_forfeit.cost, new_forfeit.time, [G.edges_list[(source.id, node.id)].id], "forfeit_edge")
                    this.add_edge(dest_node.id, sink.id, 0.0, 0.0, [0], "sink_edge")
                end
            end
        end

    elseif s==:load_transhipments
        function(G::Graph)
            for trans_node in G.transhipment_nodes
                if length(G.call_node_dict[trans_node.port]) < 3
                    call1 = G.call_node_dict[trans_node.port][1]
                    call2 = G.call_node_dict[trans_node.port][2]
                    cost, time = G.get_edge_cost_and_time(call1.id, call2.id)
                    if (call1.port ∉ this.destinations) & (call1.port != this.source_port)
                        #If in destinations, call node already added by load source and demand
                        this.add_node(call1.port, "call_node", call1.id)
                        this.add_node(call2.port, "call_node", call2.id)
                    end                    
                    this.add_edge(call1.id, call2.id, cost, time, [G.edges_list[(call1.id, call2.id)].id], "transhipment_edge")
                    this.add_edge(call2.id, call1.id, cost, time, [G.edges_list[(call2.id, call1.id)].id], "transhipment_edge")

                else
                    this.add_node(trans_node.port, "transhipment_node", trans_node.id)
                    for call_node in G.call_node_dict[trans_node.port]
                        if (call_node.port ∉ this.destinations) & (call_node.port != this.source_port)
                            #If in destinations, call node already added by load source and demand
                            this.add_node(call_node.port, "call_node", call_node.id)
                        end
                        cost, time = G.get_edge_cost_and_time(call_node.id, trans_node.id)
                        this.add_edge(call_node.id, trans_node.id, cost, time, [G.edges_list[(call_node.id, trans_node.id)].id], "transhipment_edge")
                        cost, time = G.get_edge_cost_and_time(trans_node.id, call_node.id)
                        this.add_edge(trans_node.id,call_node.id, cost, time, [G.edges_list[(trans_node.id, call_node.id)].id], "transhipment_edge")
                    end
                end
            end
        end

    
    elseif s==:load_rotations
        function(G::Graph)
            for rotation in G.rotation_list
                #find first call node that is part of sub
                rot_len = length(rotation)
                rot = 1
                first_call = rotation[rot]
                while !haskey(this.node_dict, first_call.id)
                    rot+=1
                    first_call = rotation[rot]
                end

                ################################

                #Create rotation calls
                start_ind = rot + 1
                if start_ind == rot_len+1
                    start_ind = 1
                end
                new_call = rotation[start_ind]
                
                edges_in = [G.edges_list[(first_call.id, new_call.id)].id]
                
                from = first_call.id # 
                to = new_call.id

                cost = G.edges_list[(from, to)].cost
                time = G.edges_list[(from, to)].time

                while(start_ind != rot)
            
                    if haskey(this.node_dict, new_call.id)
                        #println(from, " ", to, " has edges", edges_in)
                        this.add_edge(from, to, cost, time, edges_in, "voyage_edge")
                        start_ind += 1
                        if start_ind == rot_len+1
                            start_ind = 1
                        end
                        new_call = rotation[start_ind]
                        cost = G.edges_list[(to, new_call.id)].cost
                        time = G.edges_list[(to, new_call.id)].time
                        from = to
                        to = new_call.id
                        edges_in = [G.edges_list[(from, to)].id]
                        #println(from," ", to)
                    else
                        start_ind += 1
                        if start_ind == rot_len+1
                            start_ind = 1
                        end
                        new_call = rotation[start_ind]
                        push!(edges_in, G.edges_list[(to, new_call.id)].id)
                        cost += G.edges_list[(to, new_call.id)].cost
                        time += G.edges_list[(to, new_call.id)].time
                        to = new_call.id

                    end                    
                end
                if from == to
                    continue
                end
                #println()
                #println(from, " ", to, " has edges", edges_in)
                this.add_edge(from, to, cost, time, edges_in, "voyage_edge")



            end
        end
    elseif s==:get_summary
        function()
            println("Number of edges is ", length(collect(this.edges_list)))
            edges_stats = Dict("transhipment_edge" => 0, "load_edge" => 0, "forfeit_edge" => 0, "voyage_edge" => 0, "sink_edge" => 0)
            for (ij, edge) in this.edges_list
                edges_stats[edge.type] += 1
            end
            println("With the following distribution of edges ")
            println(edges_stats)
            println()
            println("Number of Nodes is ", length(this.node_dict))
            node_stats = Dict("transhipment_node" => 0, "source_node" => 0, "call_node" => 0, "destination_node" => 0, "sink_node" => 0)
            for (key, node) in this.node_dict
                node_stats[node.type] += 1
            end
            println("With the following distribution of nodes ")
            println(node_stats)
        end


    elseif s==:update_dual
        function(G::Graph)
        for (i, edge) in enumerate(this.edges_ordered)
            if edge.type != "sink_edge"
                dual = 0
                for edge_in_base in edge.edges_in_base
                    dual += round(G.edges_ordered[edge_in_base].lambda, digits = 7)
                end
                edge.dual = dual

            else

            end
        end
        duals = Vector{Float64}()
        max_times = Vector{Float64}()
        revs = Vector{Float64}()
        for port in this.destinations
            dual = round(G.dual_od_pair_dict[(this.source_port, port)], digits=7)
            demand = G.demand_od_path_dict[(this.source_port, port)]
            max_time = demand.transit_time
            push!(revs, demand.revenue)
            push!(duals, dual)
            this.edges_list[(G.port_dict[port].id, this.sink_id)].dual = dual

            #this.data_frame.pi_ij[this.edges_list[(G.port_dict[port].id, this.sink_id)].id] = dual
        end
        if maximum(duals+revs) <= 0
            return true, 0.0
        end
        for i in 1:length(this.data_frame.max_pj)

            this.data_frame.meanpj[i] = mean(duals)
            this.data_frame.max_pj[i] = maximum(duals)
            this.data_frame.min_pj[i] = minimum(duals)

            this.data_frame.times_chosen_for_sub[i] +=  this.data_frame.in_path[i]
            this.data_frame.in_path[i] = 0
        end
        i = 0
        for edge in this.edges_ordered
            if edge.type in ["sink_edge", "load_edge", "forfeit_edge"]
                continue
            end
            i += 1
            this.data_frame.lambda_ij[i] = edge.dual

            this.data_frame.iteration[i] += 1
            node_i = this.node_dict[edge.ij[1]]
            node_j = this.node_dict[edge.ij[2]]

            lambda_i = [edge.dual for edge in node_i.outgoing_edges]
            lambda_j = [edge.dual for edge in node_j.outgoing_edges]

            this.data_frame.arcs_lambda_i_sub_min[i] =  minimum(lambda_i)
            this.data_frame.arcs_lambda_i_sub_max[i] = maximum(lambda_i)
            this.data_frame.arcs_lambda_i_sub_mean[i] = mean(lambda_i)
            if edge.ij[2] != this.sink_id
                this.data_frame.arcs_lambda_j_sub_min[i] =  minimum(lambda_j)
                this.data_frame.arcs_lambda_j_sub_max[i] = maximum(lambda_j)
                this.data_frame.arcs_lambda_j_sub_mean[i] = mean(lambda_j)
            end

            #Ingoing
            lambda_i = [edge.dual for edge in node_i.ingoing_edges]
            lambda_j = [edge.dual for edge in node_j.ingoing_edges]

            this.data_frame.arcs_ent_lambda_i_sub_min[i] =  minimum(lambda_i)
            this.data_frame.arcs_ent_lambda_i_sub_mean[i] = mean(lambda_i)
            if edge.ij[2] != this.sink_id
                this.data_frame.arcs_ent_lambda_j_sub_min[i] =  minimum(lambda_j)
                this.data_frame.arcs_ent_lambda_j_sub_mean[i] = mean(lambda_j)
            end

        end
        this.data_frame.normalize_dyn(this)
        return false, maximum(duals+revs)
    end
    elseif s== :get_sub_features_for_node #num arcs leaving, min, max, mean co
    function(node_id::Int, G::Graph, ingoing::Bool)
        node = this.node_dict[node_id]
        base_node = G.node_list[node_id]
        if ingoing
            time = [edge.time for edge in node.outgoing_edges]
            cost = [edge.cost for edge in node.outgoing_edges]
            cap = [edge.capacity for edge in base_node.outgoing_edges if edge.type == "voyage_edge"]
            if this.node_dict[node_id].type == "transhipment_node"
                cap = [10000]
            end
            len = length(node.outgoing_edges)
        else
            time = [edge.time for edge in node.ingoing_edges]
            cost = [edge.cost for edge in node.ingoing_edges]
            cap = [edge.capacity for edge in base_node.ingoing_edges if edge.type == "voyage_edge"]
            if this.node_dict[node_id].type == "transhipment_node"
                cap = [10000]
            end
            len = length(node.ingoing_edges)
        end
        return mean(cost), maximum(cost), minimum(cost),mean(cap), maximum(cap), minimum(cap), mean(time), maximum(time), minimum(time), len
    end
    elseif s==:create_observations
        function(G::Graph, env::Env)
            id = 0
            for edge in this.edges_ordered
                if edge.type in ["load_edge", "sink_edge", "forfeit_edge"]
                    continue
                end
                id += 1
                edge.obs_id = id
                push!(this.data_frame.instance_name, env.name)
                push!(this.data_frame.in_path, 0)
                push!(this.data_frame.in_opt, 0)

                push!(this.data_frame.edge_id, edge.id)
                push!(this.data_frame.iteration, 0)
                push!(this.data_frame.sub_graph_id, this.id)

                if this.node_dict[edge.ij[1]].port in this.destinations
                    push!(this.data_frame.dest_at_i, 1)
                else
                    push!(this.data_frame.dest_at_i, 0)
                end
                if this.node_dict[edge.ij[2]].port in this.destinations
                    push!(this.data_frame.dest_at_j, 1)
                else
                    push!(this.data_frame.dest_at_j, 0)
                end

                if length(collect(this.destinations)) == 1
                    if this.node_dict[edge.ij[1]].port == collect(this.destinations)[1] 
                        push!(this.data_frame.dist_less, 0)
                    else 
                        push!(this.data_frame.dist_less, env.dist_dict[(this.node_dict[edge.ij[1]].port, collect(this.destinations)[1])] -
                        (this.node_dict[edge.ij[2]].port == collect(this.destinations)[1] ? 0 : env.dist_dict[(this.node_dict[edge.ij[2]].port, collect(this.destinations)[1])])) 
                    end
                else
                    push!(this.data_frame.dist_less, 0)
                end

                mean_cost_i, max_cost_i, min_cost_i, mean_time_i, max_time_i, min_time_i, len_base_i = G.get_base_features_for_node(edge.ij[1])
                if edge.ij[2] != this.sink_id
                    mean_cost_j, max_cost_j, min_cost_j, mean_time_j, max_time_j, min_time_j, len_base_j = G.get_base_features_for_node(edge.ij[2])
                else
                    mean_cost_j, max_cost_j, min_cost_j, mean_time_j, max_time_j, min_time_j, len_base_j = (0,0,0,0,0,0,1)
                end
                push!(this.data_frame.arc_leaving_i_base, len_base_i)
                push!(this.data_frame.arc_leaving_j_base, len_base_j)


                push!(this.data_frame.arcs_cost_i_base_min, min_cost_i)
                push!(this.data_frame.arcs_cost_i_base_max, max_cost_i)
                push!(this.data_frame.arcs_cost_i_base_mean, mean_cost_i)
                push!(this.data_frame.arcs_time_i_base_min, min_time_i)
                push!(this.data_frame.arcs_time_i_base_max, max_time_i)
                push!(this.data_frame.arcs_time_i_base_mean, mean_time_i)

                push!(this.data_frame.arcs_cost_j_base_min, min_cost_j)
                push!(this.data_frame.arcs_cost_j_base_max, max_cost_j)
                push!(this.data_frame.arcs_cost_j_base_mean, mean_cost_j)
                push!(this.data_frame.arcs_time_j_base_min, min_time_j)
                push!(this.data_frame.arcs_time_j_base_max, max_time_j)
                push!(this.data_frame.arcs_time_j_base_mean, mean_time_j)


                push!(this.data_frame.cost_of_edge, edge.cost)
                push!(this.data_frame.time_of_edge, edge.time)
                cap = 0
                for edge_base in edge.edges_in_base
                    if G.edges_ordered[edge_base].capacity >= cap
                        cap = G.edges_ordered[edge_base].capacity
                    end
                end
                push!(this.data_frame.cap_of_edge, cap)
                mean_cost_i, max_cost_i, min_cost_i,mean_cap_i, max_cap_i, min_cap_i, mean_time_i, max_time_i, min_time_i, len_sub_i = this.get_sub_features_for_node(edge.ij[1], G, false)
                if edge.ij[2] != this.sink_id
                    mean_cost_j, max_cost_j, min_cost_j,mean_cap_j, max_cap_j, min_cap_j, mean_time_j, max_time_j, min_time_j, len_sub_j = this.get_sub_features_for_node(edge.ij[2], G, false)
                else
                    mean_cost_j, max_cost_j, min_cost_j, mean_time_j, max_time_j, min_time_j, len_sub_j = (0,0,0,0,0,0,1)
                end                
                push!(this.data_frame.arc_leaving_i_sub, len_sub_i)
                push!(this.data_frame.arc_leaving_j_sub, len_sub_j)

                push!(this.data_frame.arcs_cost_i_sub_min, min_cost_i)
                push!(this.data_frame.arcs_cost_i_sub_max, max_cost_i)
                push!(this.data_frame.arcs_cost_i_sub_mean, mean_cost_i)
                push!(this.data_frame.arcs_time_i_sub_min, min_time_i)
                push!(this.data_frame.arcs_time_i_sub_max, max_time_i)
                push!(this.data_frame.arcs_time_i_sub_mean, mean_time_i)
                push!(this.data_frame.arcs_cap_i_sub_min, min_cap_i)
                push!(this.data_frame.arcs_cap_i_sub_max, max_cap_i)
                push!(this.data_frame.arcs_cap_i_sub_mean, mean_cap_i)

                push!(this.data_frame.arcs_cost_j_sub_min, min_cost_j)
                push!(this.data_frame.arcs_cost_j_sub_max, max_cost_j)
                push!(this.data_frame.arcs_cost_j_sub_mean, mean_cost_j)
                push!(this.data_frame.arcs_time_j_sub_min, min_time_j)
                push!(this.data_frame.arcs_time_j_sub_max, max_time_j)
                push!(this.data_frame.arcs_time_j_sub_mean, mean_time_j)
                push!(this.data_frame.arcs_cap_j_sub_min, min_cap_j)
                push!(this.data_frame.arcs_cap_j_sub_max, max_cap_j)
                push!(this.data_frame.arcs_cap_j_sub_mean, mean_cap_j)
                
                #Ingoing_Edges
                mean_cost_i, max_cost_i, min_cost_i,mean_cap_i, max_cap_i, min_cap_i, mean_time_i, max_time_i, min_time_i, len_sub_i = this.get_sub_features_for_node(edge.ij[1], G, true)
                if edge.ij[2] != this.sink_id
                    mean_cost_j, max_cost_j, min_cost_j,mean_cap_j, max_cap_j, min_cap_j, mean_time_j, max_time_j, min_time_j, len_sub_j = this.get_sub_features_for_node(edge.ij[2], G, true)
                else
                    mean_cost_j, max_cost_j, min_cost_j, mean_time_j, max_time_j, min_time_j, len_sub_j = (0,0,0,0,0,0,1)
                end                
                push!(this.data_frame.arc_entering_i_sub, len_sub_i)
                push!(this.data_frame.arc_entering_j_sub, len_sub_j)

                push!(this.data_frame.arcs_ent_cost_i_sub_min, min_cost_i)
                push!(this.data_frame.arcs_ent_cost_i_sub_max, max_cost_i)
                push!(this.data_frame.arcs_ent_cost_i_sub_mean, mean_cost_i)
                push!(this.data_frame.arcs_ent_time_i_sub_min, min_time_i)
                push!(this.data_frame.arcs_ent_time_i_sub_max, max_time_i)
                push!(this.data_frame.arcs_ent_time_i_sub_mean, mean_time_i)
                push!(this.data_frame.arcs_ent_cap_i_sub_min, min_cap_i)
                push!(this.data_frame.arcs_ent_cap_i_sub_max, max_cap_i)
                push!(this.data_frame.arcs_ent_cap_i_sub_mean, mean_cap_i)

                push!(this.data_frame.arcs_ent_cost_j_sub_min, min_cost_j)
                push!(this.data_frame.arcs_ent_cost_j_sub_max, max_cost_j)
                push!(this.data_frame.arcs_ent_cost_j_sub_mean, mean_cost_j)
                push!(this.data_frame.arcs_ent_time_j_sub_min, min_time_j)
                push!(this.data_frame.arcs_ent_time_j_sub_max, max_time_j)
                push!(this.data_frame.arcs_ent_time_j_sub_mean, mean_time_j)
                push!(this.data_frame.arcs_ent_cap_j_sub_min, min_cap_j)
                push!(this.data_frame.arcs_ent_cap_j_sub_max, max_cap_j)
                push!(this.data_frame.arcs_ent_cap_j_sub_mean, mean_cap_j)
                
                
                push!(this.data_frame.voy_edge, edge.type == "voyage_edge" ? 1 : 0)
                push!(this.data_frame.trans_edge, edge.type == "transhipment_edge" ? 1 : 0)
                push!(this.data_frame.load_edge, edge.type == "load_edge" ? 1 : 0)
                push!(this.data_frame.forfeit_edge, edge.type == "forfeit_edge" ? 1 : 0)
                push!(this.data_frame.sink_edge, edge.type == "sink_edge" ? 1 : 0)


                push!(this.data_frame.num_dest, length(this.destinations))
                push!(this.data_frame.lambda_ij, 0.0)

                push!(this.data_frame.arcs_lambda_i_sub_min , 0.0)
                push!(this.data_frame.arcs_lambda_i_sub_max , 0.0)
                push!(this.data_frame.arcs_lambda_i_sub_mean, 0.0)
                push!(this.data_frame.arcs_lambda_j_sub_min , 0.0)
                push!(this.data_frame.arcs_lambda_j_sub_max , 0.0)
                push!(this.data_frame.arcs_lambda_j_sub_mean, 0.0)
                push!(this.data_frame.arcs_ent_lambda_i_sub_min , 0.0)
                push!(this.data_frame.arcs_ent_lambda_i_sub_mean, 0.0)
                push!(this.data_frame.arcs_ent_lambda_j_sub_min , 0.0)
                push!(this.data_frame.arcs_ent_lambda_j_sub_mean, 0.0)


                push!(this.data_frame.pi_ij, 0.0)
                push!(this.data_frame.times_chosen_for_sub, 0.0)
                push!(this.data_frame.meanpj, 0.0)
                push!(this.data_frame.max_pj, 0.0)
                push!(this.data_frame.min_pj, 0.0) 

            end
        end

    elseif s==:build_subgraph
        function(G::Graph, env)
            this.load_source_and_demand(G)
            this.load_transhipments(G)
            this.load_rotations(G)
            this.create_observations(G, env)
            #this.data_frame.normalize(this)
            #this.get_summary()
        end
    elseif s==:predict
        function(model)
            if this.static
                preds = model(obs_to_floats(this.data_frame))
            else
                preds = model(obs_to_floats_dynamic(this.data_frame))
            end 
            probs = softmax(preds)
            for (i,edge) in enumerate(this.edges_ordered)
                if edge.type in ["load_edge", "sink_edge", "forfeit_edge"]
                    continue
                end
                edge.prob = probs[1, edge.obs_id]
                #println(edge.prob)
                
            end  
        end

    else getfield(this, s)

    end
end

function init_sub_graph_o_all(G::Graph, source::String, env::Env, sink_id::Int, id::Int)
    destinations = []
    max_trans_time = 0
    max_rev = 0
    base_graph_node = G.port_dict[source]
    total_demand = 0
    for dest in env.demand_dict[source]
        push!(destinations, dest.destination)
        total_demand += dest.ffe_per_week
        if dest.revenue >= max_rev
            max_rev = dest.revenue
        end
        if dest.transit_time >= max_trans_time
            max_trans_time = dest.transit_time
        end

    end
    dataframe = Observations(Vector{Int}(),Vector{Int}(), Vector{String}(), Vector{Int}(), Vector{Int}(), Vector{Int}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), 
                Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(),Vector{Float64}())
    return GraphSub(id, Dict{Tuple{Int64, Int64}, EdgeSub}(),Vector{EdgeSub}(), Dict{Int, NodeSub}(), source, base_graph_node.id, Set(destinations), 0, max_trans_time, sink_id, dataframe, false, true, total_demand, 0.0, 0, max_rev)
end


function init_sub_graph_o_d(G::Graph, source::String, env::Env, sink_id::Int, id::Int, demand::Demand)
    destinations = [demand.destination]
    max_trans_time = demand.transit_time
    total_demand = demand.ffe_per_week
    base_graph_node = G.port_dict[source]

    dataframe = Observations(Vector{Int}(),Vector{Int}(), Vector{String}(), Vector{Int}(), Vector{Int}(), Vector{Int}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), 
                Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
                , Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(),Vector{Float64}())
    return GraphSub(id, Dict{Tuple{Int64, Int64}, EdgeSub}(),Vector{EdgeSub}(), Dict{Int, NodeSub}(), source, base_graph_node.id, Set(destinations), 0, max_trans_time, sink_id, dataframe, false, true, total_demand, 0.0, 0, 0.0)
end






