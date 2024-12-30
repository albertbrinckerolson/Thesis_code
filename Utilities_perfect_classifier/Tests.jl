
include("complete_collumn_gen_o_d.jl")
include("complete_collumn_gen_oall.jl")

main_o_all(["Demand_EuropeAsia"], ["EuropeAsia_pid_26674_12"], "train_df", true)
main_od(["Demand_EuropeAsia"], ["EuropeAsia_pid_26674_12"], "train_df", true)


println(objective_value(master))
println(objective_value(master2))
function test()
    return
end



G.demand_od_path_dict
demand_cound = Dict(key => Vector{Path}() for (key, value) in G.demand_od_path_dict)
demand_cound2 = Dict(key => Vector{Path}() for (key, value) in G.demand_od_path_dict)
for path in all_paths
    push!(demand_cound[(path.origin, path.destination)], path)
end
for path in all_paths2
    push!(demand_cound2[(path.origin, path.destination)], path)
end
not_found = []

for (key, value) in demand_cound
    println(length(value), " ", length(demand_cound2[key]))
    if length(value) != length(demand_cound2[key])
        push!(not_found, value)
        push!(not_found, demand_cound2[key])
        println("FUCK")
    end
end



not_found2 = []
it = 0
for path in all_paths2
    it+=1
    found = false
    for path2 in all_paths
        if path.edges == path2.edges
            found = true
            if path.voy_edges != path2.voy_edges
                found = false
            end
            if path.demand_id != path2.demand_id
                found = false
            end
        end
    end
    if found == false
        push!(not_found2, path)
    end
    println(it)
end

not_found1 = []
for path in all_paths
    it+=1
    found = false
    for path2 in all_paths2
        if path.edges == path2.edges
            found = true
            if path.voy_edges != path2.voy_edges
                found = false
            end
            if path.demand_id != path2.demand_id
                found = false
            end
        end
    end
    if found == false
        push!(not_found1, path)
    end
    println(it)
end



for path_list in not_found
    for path in path_list
        println(path.origin)
        println(path.edges)
        println(path.demand_id)
    end
end


for path in not_found2
    println("Path ", path.edges)
    sub_we_want_oall = 0
    for sub in list_of_sub_oall
        if sub.source_port == path.origin
            sub_we_want_oall = sub
        end
    end

    sub_we_want_od = 0
    for sub in list_of_sub_od
        if (collect(sub.destinations)[1] == path.destination) & (sub.source_port == path.origin)
            sub_we_want_od = sub
        end
    end

    for edge in path.edges[1:end]
        println(G.edges_ordered[edge])
        println(sub_we_want_od.edges_list[G.edges_ordered[edge].ij])
        println(sub_we_want_oall.edges_list[G.edges_ordered[edge].ij])

        
    end




end





sub_we_want_oall.source_port
sub_we_want_od.source_port
"YEADE" in sub_we_want_oall.destinations
sub_we_want_od.destinations
G.port_dict["YEADE"]
sub_we_want_od.node_dict[95]
sub_we_want_oall.node_dict[95]

sub_we_want_oall.get_summary()
sub_we_want_od.get_summary()

for (key, value) in sub_we_want_oall.node_dict
    if haskey(sub_we_want_od.node_dict, key)
        println(value)
        println(value.type)
    end
end

for path in not_found1
    i = G.edges_ordered[path.edges[1]].ij[1]
    j = G.edges_ordered[path.edges[1]].ij[2]

end


path = not_found1[1]

for edge in reverse(path.edges)
    println(G.edges_ordered[edge].ij)
    println(edge)
    println(sub_we_want_od.edges_list[G.edges_ordered[edge].ij])

    println(sub_we_want_oall.edges_list[G.edges_ordered[edge].ij])
end