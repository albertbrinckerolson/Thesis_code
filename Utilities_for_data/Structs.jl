
include("ReadLogs.jl")

mutable struct Edge
    id::Int64
    ij :: Tuple{Int64, Int64}
    cost :: Float64
    time :: Float64
    capacity :: Float64
    lambda :: Float64
    paths_using_edge :: Vector{Int64}
    type::String
end


mutable struct Node
    id :: Int64
    port :: String
    outgoing_edges :: Vector{Edge}
    ingoing_edges :: Vector{Edge}
    type::String
end


mutable struct Graph
    call_node_dict :: Dict{String, Vector{Node}}
    port_dict :: Dict{String, Node}
    edges_list :: Dict{Tuple{Int64, Int64}, Edge}
    edges_ordered :: Vector{Edge}
    node_list :: Vector{Node}
    rotation_list :: Vector{Vector{Node}}
    source_node_id :: Int64
    destinations :: Vector{Int64}
    number_of_nodes :: Int64
    number_of_edges :: Int64
    transhipment_nodes :: Vector{Node}
    num_paths :: Int
    demand_od_path_dict :: Dict{Tuple{String,String}, Demand}
    dual_od_pair_dict :: Dict{Tuple{String, String}, Float64}
    voy_edge_start_id ::Int
    voy_edge_end_id ::Int
end


function Base.getproperty(this::Graph, s::Symbol)
    if s == :add_node 
        function(port::String, type::String)
            """
            Adds node to graph. If it is a call node it appends it to the the ports call node dictionary
            It asumes that the firstly added ports are the port nodes

            """
            this.number_of_nodes += 1
            new_node = Node(this.number_of_nodes, port, Vector{Edge}(),Vector{Edge}(), type)
            push!(this.node_list, new_node)
            if !haskey(this.call_node_dict, port)
                this.call_node_dict[port] = Vector{Node}()
                this.port_dict[port] = new_node
            end
            if type == "call_node"
                push!(this.call_node_dict[port], new_node) 
            end
            return new_node
        end

    elseif  s == :add_edge

        function(node1id::Int64, node2id::Int64, cost::Float64, time::Float64, capacity:: Float64, double::Bool, type::String)
            """
            Function that adds an edge with a given cost time and capacity. possible to double connect with double::Bool in case of laod edges
            If it is a transhipment edge it will add a 0-cost, 0-time edge the the from call to trans node.
            """
            this.number_of_edges += 1
            if (type == "transhipment_edge") & double
                cost = cost/2
                time = time/2
            end
            new_edge = Edge(this.number_of_edges, (node1id, node2id), cost, time, capacity, 0, Vector{Int64}(), type)
            push!(this.edges_ordered, new_edge)
            this.edges_list[new_edge.ij] = new_edge
            push!(this.node_list[node1id].outgoing_edges, new_edge)
            push!(this.node_list[node2id].ingoing_edges, new_edge)

            if double 
                if type == "transhipment_edge"
                    this.add_edge(node2id, node1id,  cost, time, capacity, false, type)
                else
                    this.add_edge(node2id, node1id, cost, time, capacity, false, type)
                end
            end
        end


    elseif s == :update_lambda
        function(lambda :: Vector{Float64})
            """
            Function to update lambda given the duals
            """
            for (i,edge) in enumerate(this.edges_ordered[this.voy_edge_start_id:this.voy_edge_end_id])
                edge.lambda = lambda[i]
            end
        end



    elseif s == :load_ports
        function(env::Env)
            """
            Loads all ports and connect them all for forfeit nodes

            """
            ports = get_unique_ports(env.rotations)
            #Adds port nodes to graph
            for port in ports
                this.add_node(port, "port_node")
            end
            #Adds forfeit edges
            port_tuple_list = collect(this.port_dict) 
            #for (i, (port1_name, port1_node)) in  enumerate(port_tuple_list) 
            #    if haskey(env.demand_dict,port1_name)
            #        for demand in env.demand_dict[port1_name]
            #            this.add_edge(port1_node.id, this.port_dict[demand.destination].id, 1000.0, 0.0, env.bigM, false, "forfeit_edge")
            #        end
            #    end
            #end
        end
    elseif s == :load_rotations
        """
        Loads rotations and create the resulting call nodes
        """
        function(env::Env)
            this.voy_edge_start_id = this.number_of_edges + 1
            for rotation in env.rotations

                first_port = rotation.path[1]
                current_call = this.add_node(first_port, "call_node")
                rotation_list = [current_call]
                first_id = current_call.id

                for port2 in rotation.path[2:end]

                    port2_call = this.add_node(port2, "call_node")
                    push!(rotation_list, port2_call)

                    cost, time, capacity = calculate_edge_data(current_call.port, port2_call.port, "voyage", rotation, env)
                    this.add_edge(current_call.id, port2_call.id, cost, time, capacity, false, "voyage_edge")

                    current_call = port2_call
                end
                
                cost, time, capacity = calculate_edge_data(current_call.port, first_port, "voyage", rotation, env)
                

                this.add_edge(current_call.id, first_id, cost, time, capacity, false,"voyage_edge")
                push!(this.rotation_list, rotation_list)

            end
            this.voy_edge_end_id = this.number_of_edges
        end
    elseif s == :load_trans_and_load_edges_star
        function(env::Env)
            for (port, port_node) in this.port_dict
                call_nodes = this.call_node_dict[port]
                if length(call_nodes)==1
                    cost, time, capacity = calculate_edge_data(call_nodes[1].port, call_nodes[1].port, "load", env.rotations[1], env)
                    this.add_edge(port_node.id, call_nodes[1].id, cost, time, capacity, true, "load_edge")
                else
                    trans_ship_node = this.add_node(port, "transhipment_node")
                    push!(this.transhipment_nodes, trans_ship_node)
                    if length(call_nodes) < 3 
                        cost, time, capacity = calculate_edge_data(port, port, "trans", env.rotations[1], env)
                        this.add_edge(call_nodes[1].id, call_nodes[2].id, cost,time,capacity, false, "transhipment_edge")
                        this.add_edge(call_nodes[2].id, call_nodes[1].id, cost,time,capacity, false, "transhipment_edge")
                        
                        cost, time, capacity = calculate_edge_data(port, port, "load", env.rotations[1], env)
                        
                        this.add_edge(port_node.id, call_nodes[1].id, cost, time, capacity, true, "load_edge")
                        this.add_edge(port_node.id, call_nodes[2].id, cost, time, capacity, true, "load_edge")

                    else
                        for call_node in call_nodes
                            cost, time, capacity = calculate_edge_data(port, port, "trans", env.rotations[1], env)
                            this.add_edge(trans_ship_node.id, call_node.id, cost,time,capacity, true, "transhipment_edge")
                            
                            cost, time, capacity = calculate_edge_data(port, port, "load", env.rotations[1], env)
                            this.add_edge(port_node.id, call_node.id, cost, time, capacity, true, "load_edge")
                        end
                    end
                end

            end
        end

    elseif s == :get_summary_statistics
        function()
            println("Number of edges is ", length(collect(this.edges_list)))
            edges_stats = Dict("transhipment_edge" => 0, "load_edge" => 0, "forfeit_edge" => 0, "voyage_edge" => 0)
            for (ij, edge) in this.edges_list
                edges_stats[edge.type] += 1
            end
            println("With the following distribution of edges ")
            println(edges_stats)
            println()
            println("Number of Nodes is ", length(this.node_list))
            node_stats = Dict("transhipment_node" => 0, "port_node" => 0, "call_node" => 0)
            for node in this.node_list
                node_stats[node.type] += 1
            end
            println("With the following distribution of nodes ")
            println(node_stats)


        end
    elseif s==:construct_od_dict
        function(env::Env)
            demand_od = Vector{Tuple{String, String}}()
            demands = Vector{Demand}()
            for (key, demand_list) in env.demand_dict
                for demand in demand_list
                    dest = demand.destination
                    i = 0
                    old_dem = 0
                    more_demand = false
                    while haskey(this.demand_od_path_dict, (key, dest))
                        old_dem = this.demand_od_path_dict[(key, demand.destination)]
                        i+=1
                        dest = demand.destination*string(i)
                        more_demand = true
                    end
                    if (more_demand) 
                        if  (old_dem.transit_time < demand.transit_time) 
                            this.demand_od_path_dict[(key, demand.destination)] = demand
                            this.demand_od_path_dict[(key, dest)] = old_dem
                            this.dual_od_pair_dict[(key, dest)] = 0.0       
                        else
                            this.demand_od_path_dict[(key, dest)] = demand 
                            this.dual_od_pair_dict[(key, dest)] = 0.0    
                        end
                    else
                        this.demand_od_path_dict[(key, dest)] = demand 
                        this.dual_od_pair_dict[(key, dest)] = 0.0       
                    end    
                end
            end
        end

    elseif s== :update_pi
        function(pi_duals)
            for (i, (key, old)) in enumerate(this.dual_od_pair_dict)
                this.dual_od_pair_dict[key] = pi_duals[this.demand_od_path_dict[key].id]
            end
        end

    
    elseif s== :build_star_graph
        function(env::Env)

            this.load_ports(env)
            this.load_rotations(env)
            this.load_trans_and_load_edges_star(env)
            this.construct_od_dict(env)
            this.get_summary_statistics()
            
        
        end
    elseif s== :get_edge_cost_and_time
        function(node1id::Int, node2id::Int)
            cost = this.edges_list[(node1id, node2id)].cost
            time = this.edges_list[(node1id, node2id)].time
            return cost, time
        end
    elseif s== :get_base_features_for_node #num arcs leaving, min, max, mean co
        function(node_id::Int)
            node = this.node_list[node_id]
            time = [edge.time for edge in node.outgoing_edges]
            cost = [edge.cost for edge in node.outgoing_edges]

            len = length(node.outgoing_edges)
            return mean(cost), maximum(cost), minimum(cost), mean(time), maximum(time), minimum(time), len
        end
    
    else getfield(this, s)

    end
end

