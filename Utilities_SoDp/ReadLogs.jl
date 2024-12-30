using Pkg, CSV, DataFrames,Serialization





struct Rotation
    id :: Int64
    path :: Vector{String}
    speed :: Float64
    vessel_type_cap :: Int64
    number_of_vessels :: Int64
    butterfly :: Bool
    voyage_distance :: Float64
    voyage_duration :: Float64
    port_call_cost :: Float64
    bunker_idle_burn :: Float64
    bunker_fuel_burn :: Float64
    total_bunker_cost :: Float64
    total_tc_cost :: Float64
    weekly_capacity :: Float64
end


struct Demand
    id::Int
    origin::String
    destination::String 
    ffe_per_week::Float64
    revenue:: Float64
    transit_time :: Float64
end

struct Port
    unlocode::String
    name::String
    cost_per_full::Float64
    cost_per_full_trnsf::Float64
end

struct Env
    name :: String
    port_dict :: Dict{String,Port}
    dist_dict :: Dict{Tuple{String,String},Float64}
    bigM :: Float64
    trans_time :: Float64
    load_time :: Float64
    demand_dict :: Dict{String, Vector{Demand}}
    demand_list :: Vector{Demand}
    rotations :: Vector{Rotation}
end

function parse_txt_rotations(filename)
    solution = include("Instance-WL.txt")
    rotations = Vector{Rotation}()
    vessel_cap_dict = Dict( 1 =>    450,
                            2 =>    800,
                            3 =>    1200,
                            4 =>    2400,
                            5 =>    4200,
                            6 =>    7500)

    ports_df = CSV.read("LINERLIB-master/data/ports.csv", DataFrame, ntasks=1)
    df_sorted = sort(ports_df, :UNLocode)
    df_sorted_2 = filter(row -> (row.UNLocode in codes),df_sorted)


    for (i,route) in enumerate(solution.Rotations)
        id = i
        path = df_sorted_2[!,"UNLocode"][route.Calls]
        type_cap = vessel_cap_dict[route.Vessel]
        weekly_capacity = (route.nVessels * type_cap) / route.Duration
        
        # DONE LIKE THIS IN READ LOGS -> current_weekly_capacity = (current_number_of_vessels * current_vessel_type_cap) / current_voyage_duration
        push!(rotations,Rotation( id,
                        path,
                        route.Speed,
                        type_cap,
                        route.nVessels,
                        false,
                        route.Distance,
                        route.Duration,
                        0.0,
                        0.0,
                        0.0, 
                        0.0,
                        0.0,
                        weekly_capacity))
    end
    return rotations
end

# Function to parse the log file and extract rotation data
function parse_rotations(file_path::String)
    rotations = Vector{Rotation}()
    
    current_rotation_id = nothing
    current_path = String[]
    current_speed = 0.0
    current_vessel_type_cap = 0
    current_number_of_vessels = 0
    current_butterfly = false
    current_voyage_distance = 0.0
    current_voyage_duration = 0.0
    current_port_call_cost = 0.0
    current_bunker_idle_burn = 0.0
    current_bunker_fuel_burn = 0.0
    current_total_bunker_cost = 0.0
    current_total_tc_cost = 0.0
    current_weekly_capacity = 0.0

    # Read the log file line by line
    open(file_path, "r") do file
        for line in eachline(file)
            if contains(line,"------")
                break
            end
            if startswith(line, "service")
                if !isnothing(current_rotation_id)
                    # Save the previous rotation before starting a new one
                    current_weekly_capacity = (current_number_of_vessels * current_vessel_type_cap) / current_voyage_duration
                    push!(rotations, Rotation(
                        current_rotation_id, current_path, current_speed, current_vessel_type_cap, 
                        current_number_of_vessels, current_butterfly, current_voyage_distance, 
                        current_voyage_duration, current_port_call_cost, current_bunker_idle_burn, 
                        current_bunker_fuel_burn, current_total_bunker_cost, current_total_tc_cost,
                        current_weekly_capacity 

                    ))
                end
                
                # Reset for the new rotation
                current_rotation_id = parse(Int64, split(line)[end])
                current_path = String[]
                current_speed = 0.0
                current_vessel_type_cap = 0
                current_number_of_vessels = 0
                current_butterfly = false
                current_voyage_distance = 0.0
                current_voyage_duration = 0.0
                current_port_call_cost = 0.0
                current_bunker_idle_burn = 0.0
                current_bunker_fuel_burn = 0.0
                current_total_bunker_cost = 0.0
                current_total_tc_cost = 0.0
            elseif contains(line, "capacity")
                current_vessel_type_cap = parse(Int64, split(line)[end])
            elseif contains(line, "# vessels")
                current_number_of_vessels = parse(Int64, split(line)[end])
            elseif contains(line, "Butterfly rotation")
                current_butterfly = true
            elseif contains(line, "speed")
                current_speed = parse(Float64, split(line)[end])
            elseif contains(line, "voyage distance")
                current_voyage_distance = parse(Float64, split(line)[end])
            elseif contains(line, "voyage duration")
                current_voyage_duration = parse(Float64, split(line)[end])
            elseif contains(line, "Port call cost")
                current_port_call_cost = parse(Float64, split(line)[end])
            elseif contains(line, "Bunker idle burn")
                current_bunker_idle_burn = parse(Float64, split(line)[end])
            elseif contains(line, "Bunker fuel burn")
                current_bunker_fuel_burn = parse(Float64, split(line)[end])
            elseif contains(line, "Total bunker cost")
                current_total_bunker_cost = parse(Float64, split(line)[end])
            elseif contains(line, "Total TC cost")
                current_total_tc_cost = parse(Float64, split(line)[end])
            elseif length(split(line)) >= 3 && isdigit(line[1])
                port_string = split(line)[2]
                push!(current_path, port_string)
            end
        end
        
        # Save the last rotation after the loop ends
        if !isnothing(current_rotation_id)
            current_weekly_capacity = (current_number_of_vessels * current_vessel_type_cap) / current_voyage_duration
            push!(rotations, Rotation(
                current_rotation_id, current_path, current_speed, current_vessel_type_cap, 
                current_number_of_vessels, current_butterfly, current_voyage_distance, 
                current_voyage_duration, current_port_call_cost, current_bunker_idle_burn, 
                current_bunker_fuel_burn, current_total_bunker_cost, current_total_tc_cost,
                current_weekly_capacity 
            ))
        end
    end
    
    return rotations
end

function read_demand(path::String, o_all:: Bool, ports::Set{String}, scale_time_add::Float64, scale_demand_mul::Float64)
    demand_df = CSV.read(path, DataFrame, ntasks=1)
    o_all = Dict(collect(ports)[i] => Vector{Demand}() for i in 1:length(ports))
    id = 1
    demand_list = Vector{Demand}()
    demand_origin_dest_pairs = Set{Tuple{String, String}}()
    for row in eachrow(demand_df)
        if haskey(o_all,row.Origin) 
            if row.Destination in ports
                new_demand = Demand(id,row.Origin, row.Destination, row.FFEPerWeek*scale_demand_mul, row.Revenue_1, row.TransitTime + scale_time_add)
                
                push!(demand_list, new_demand)
                push!(o_all[row.Origin], new_demand)
                id+=1
            end
        end
    end
    return o_all, demand_list
end

function get_unique_ports(rotations::Vector{Rotation})
    unique_ports = Vector{String}()
    for r in rotations
        append!(unique_ports, r.path)
    end
    return Set(unique_ports)
end

function create_env(bigM,trans_time,load_time, instance_demand::String, instance_rotation, scale_time_add=0.0, scale_demand_mul=1.0)

    ports_path = "LINERLIB-master/data/ports.csv"
    ports_df = CSV.read(ports_path, DataFrame,types=Dict(i=>Float64 for i in [9 10 11 12]), ntasks=1)
    
    # Check for rows with any missing values
    has_missing = [any(ismissing, row) for row in eachrow(ports_df)]

    #df_with_missing = df[has_missing, :]
    df_without_missing = ports_df[.!has_missing, :]
    #ports_with_missing = Dict(row.UNLocode => Port(row.UNLocode,row.name,row.CostPerFULL,row.CostPerFULLTrnsf,row.PortCallCostFixed) for row in eachrow(df_with_missing))
    port_dict = Dict(row.UNLocode => Port(row.UNLocode,row.name,row.CostPerFULL,row.CostPerFULLTrnsf) for row in eachrow(df_without_missing))

    distance_path = "LINERLIB-master/data/dist_dense.csv" # Replace with the actual file path
    dist_df = CSV.read(distance_path, DataFrame, ntasks=1)
    dist_dict = Dict((row.fromUNLOCODe, row.ToUNLOCODE) => row.Distance for row in eachrow(dist_df))
    rotations = nothing
    if endswith(instance_rotation,"txt")
        rotations  = deserialize("WL_rotations.bin")
    else
        rotations = parse_rotations("LINERLIB-master/results/BrouerDesaulniersPisinger2014/"*instance_rotation*".log")
    end

    #rotations = parse_rotations("LINERLIB-master/results/BrouerDesaulniersPisinger2014/"*instance_rotation*".log")
    ports = get_unique_ports(rotations)

    path_demand = "LINERLIB-master/data/"*instance_demand*".csv"
    demand_dict, demand_list = read_demand(path_demand, true, ports, scale_time_add, scale_demand_mul)

    return Env(instance_rotation, port_dict,dist_dict,bigM, trans_time, load_time, demand_dict, demand_list, rotations)
end 


function calculate_edge_data(port_i:: String, port_j::String, type_of_edge::String,rotation::Rotation,env::Env) :: Tuple{Float64,Float64,Float64}
    if type_of_edge == "voyage"
        cap = rotation.weekly_capacity
        travel_time = (env.dist_dict[(port_i,port_j)] / rotation.speed) / 24 
        cost = 0.0
    elseif type_of_edge == "trans"
        cap = env.bigM
        travel_time = env.trans_time
        cost = env.port_dict[port_i].cost_per_full_trnsf
    elseif type_of_edge == "load"
        cap = env.bigM
        travel_time = env.load_time
        cost = env.port_dict[port_i].cost_per_full
    end 

    return cost, travel_time, cap
end


