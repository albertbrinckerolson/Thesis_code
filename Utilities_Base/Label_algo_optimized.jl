include("ReadLogs.jl")
include("Structs.jl")
include("Struct_sub_graph.jl")

using DataStructures

struct Label
    id :: Int64
    visited :: Set{Int64}
    current_node :: Int64
    accumulated_time :: Float64
    accumulated_cost :: Float64
    real_cost :: Float64
    parent_label_id :: Int64
end

struct Path
    sub_id:: Int
    sub_edges :: Vector{Int}
    edges :: Vector{Int}
    voy_edges :: Vector{Int}
    cost :: Float64
    destination :: String
    origin :: String
    demand_id :: Int
end


function Base.getproperty(this::Label, s::Symbol)
    if s == :dominates
        function(other_label::Label)
            """
            Checks if label is dominated by another. L1.dominates(L2) will return true if L1 dominates L2
            """
        if   (this.current_node == other_label.current_node) &
            (this.accumulated_cost ≤ other_label.accumulated_cost) &
            (this.accumulated_time ≤ other_label.accumulated_time)
            return true
        else
            return false
        end
    end
    else
        getfield(this, s)
    end
end

function backtrack!(label::Label, label_dict::Dict{Int,Label}, Gs::GraphSub, G::Graph)
    """

    """
    # nice to have a list of edges for each path
    path = Vector{Int64}()
    voy_edge = Vector{Int64}()
    previous_node = label.current_node
    destination = Gs.node_dict[previous_node].port
    current_label = label_dict[label.parent_label_id]
    sub_path = Vector{Int}()
    while current_label.current_node != Gs.source_base_id
        push!(sub_path, Gs.edges_list[(current_label.current_node, previous_node)].id)
        for edge_id in reverse(Gs.edges_list[(current_label.current_node, previous_node)].edges_in_base)
            push!(path, edge_id)
            if G.edges_ordered[edge_id].type == "voyage_edge"
                push!(voy_edge, edge_id - G.voy_edge_start_id+1)
            end
        end
        previous_node = current_label.current_node
        current_label = label_dict[current_label.parent_label_id]

    end

    # last itteration where source node is current_label.current_node
    #Gs.data_frame.in_path[Gs.edges_list[(current_label.current_node, previous_node)].id] = 1
    push!(sub_path, Gs.edges_list[(current_label.current_node, previous_node)].id)
    for edge_id in reverse(Gs.edges_list[(current_label.current_node,previous_node)].edges_in_base)
        push!(path,edge_id)
        if G.edges_ordered[edge_id].type == "voyage_edge"
            push!(voy_edge, edge_id- G.voy_edge_start_id+1)
        end
    end
    return sub_path, path, voy_edge, destination
end



function extend!(label_queue::PriorityQueue{Label,Float64}, edge::EdgeSub,num_labels, label::Label, Gs::GraphSub)

    """
    Takes a label and returns all feasible extensions. Returns updated numlabels to account for labels created in this function.
    """
        if (label.accumulated_time + edge.time <= Gs.max_time) & (edge.ij[2] ∉ label.visited)

                new_visited = push!(copy(label.visited),edge.ij[2])
                new_label = Label(num_labels,
                                new_visited, edge.ij[2],
                                edge.time + label.accumulated_time,
                                edge.cost - edge.dual + label.accumulated_cost,
                                edge.cost + label.real_cost,
                                label.id)
                enqueue!(label_queue,new_label,new_label.accumulated_time)
    end
end

function process_edge(edge::EdgeSub, G::Graph, Gs::GraphSub, path_list::Vector{Path}, label::Label, label_queue::PriorityQueue{Label,Float64}, label_dict::Dict{Int64,Label}, num_labels::Int, max_dual::Float64)
    new_labels_num = Int(0)
    rev = 0
    num_found = 0
    if (edge.ij[2] == Gs.sink_id)
        rev = G.demand_od_path_dict[Gs.source_port, Gs.node_dict[label.current_node].port].revenue
        if  (label.accumulated_cost - edge.dual -rev + 0.000001 < 0)
            demand = G.demand_od_path_dict[Gs.source_port, Gs.node_dict[label.current_node].port]
            if  demand.transit_time >= label.accumulated_time
                Gs.data_frame.in_path[Gs.edges_list[(label.current_node, Gs.sink_id)].id] = 1
                sub_path, path,voy_path, destination = backtrack!(label, label_dict, Gs, G)
                push!(path_list, Path(Gs.id, sub_path, path,voy_path, label.real_cost, destination, Gs.source_port, demand.id))
                num_found += 1
                i=1
                while haskey(G.demand_od_path_dict,(Gs.source_port, Gs.node_dict[label.current_node].port*string(i)))
                    demand = G.demand_od_path_dict[(Gs.source_port, Gs.node_dict[label.current_node].port*string(i))]
                    dual = G.dual_od_pair_dict[(Gs.source_port, Gs.node_dict[label.current_node].port*string(i))]
                    rev = G.demand_od_path_dict[(Gs.source_port, Gs.node_dict[label.current_node].port*string(i))].revenue

                    if (demand.transit_time >= label.accumulated_time) &  (label.accumulated_cost - dual - rev + 0.000001 < 0)
                        push!(path_list, Path(Gs.id, sub_path, path,voy_path, label.real_cost, destination, Gs.source_port, demand.id))
                    end
                    i+=1
                end
            end
        elseif  (label.accumulated_cost - edge.dual-rev + 0.000001 >= 0)
            #println(label.accumulated_cost - edge.dual)
            #println("Dominated")
        end
    elseif (label.accumulated_cost - max_dual + 0.000001 >= 0)
    else
        new_labels_num += 1
        extend!(label_queue, edge, num_labels+new_labels_num, label, Gs)
    end
    return new_labels_num, num_found
end




function semi_stoc_labeling_algorithm_opt!(G::Graph, path_list::Vector{Path}, Gs::GraphSub, threshold::Float64, choice_function::Function, max_dual::Float64)
    """
    This function performs the entire labelling algorithm

    """
    num_labels = 1
    num_paths = 0

    origin = Gs.source_port

    # priority queue to supply labels based on lowest accumulated time. Initialized with source node of subproblem as first label.
    label_queue = PriorityQueue{Label,Float64}(Label(1,Set{Int}(), Gs.source_base_id, 0, 0, 0, 0) => 0)

    # dict indexed by node_id to quickly access all labels with identical current node
    node_label_dict = [Vector{Label}() for id in 1:G.number_of_nodes]

    # dict to quickly access label based on label id (for backtracking)
    label_dict = Dict{Int64,Label}()

    while isempty(label_queue) == false
        label = dequeue!(label_queue)
        # check if the label is dominated
        dominated = false
        for potential_dominator in node_label_dict[label.current_node]
            if potential_dominator.dominates(label)
                dominated = true
                break
            end
        end
        if dominated
            continue
        else
            label_dict[label.id] = label
            push!(node_label_dict[label.current_node],label)

            no_probable = true
            for edge in Gs.node_dict[label.current_node].outgoing_edges
                # check if current edge is connecting path to sink
                labels, paths = process_edge(edge, G, Gs, path_list, label, label_queue, label_dict,num_labels, max_dual)
                num_labels += labels
                num_paths += paths

            end
           
        end
    end

end


function semi_stoc_labeling_algorithm_opt!_partial(G::Graph, path_list::Vector{Path}, Gs::GraphSub, threshold::Float64, choice_function::Function, max_dual::Float64, max_paths)
    """
    This function performs the entire labelling algorithm

    """
    num_labels = 1
    num_paths = 0

    origin = Gs.source_port

    # priority queue to supply labels based on lowest accumulated time. Initialized with source node of subproblem as first label.
    label_queue = PriorityQueue{Label,Float64}(Label(1,Set{Int}(), Gs.source_base_id, 0, 0, 0, 0) => 0)

    # dict indexed by node_id to quickly access all labels with identical current node
    node_label_dict = [Vector{Label}() for id in 1:G.number_of_nodes]

    # dict to quickly access label based on label id (for backtracking)
    label_dict = Dict{Int64,Label}()

    while isempty(label_queue) == false
        label = dequeue!(label_queue)
        # check if the label is dominated
        dominated = false
        for potential_dominator in node_label_dict[label.current_node]
            if potential_dominator.dominates(label)
                dominated = true
                break
            end
        end
        if dominated
            continue
        else
            label_dict[label.id] = label
            #println(label)
            push!(node_label_dict[label.current_node],label)

            no_probable = true
            for edge in Gs.node_dict[label.current_node].outgoing_edges

                # check if current edge is connecting path to sink
                labels, paths = process_edge(edge, G, Gs, path_list, label, label_queue, label_dict,num_labels, max_dual)
                num_labels+= labels
                num_paths += paths
                if max_paths < num_paths
                    return
                end


            end
        end
    end
    return
end
