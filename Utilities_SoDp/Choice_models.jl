
function random_choice_one_draw(old_vec::Vector{EdgeSub}, threshold :: Float64)
    new_vec = Vector{EdgeSub}()
    draw = rand()
    for edge in old_vec
        if edge.prob >= draw
            push!(new_vec, edge)
        end
    end
    return new_vec
end

function choose_10p_less(old_vec::Vector{EdgeSub}, threshold :: Float64)
    new_vec = Vector{EdgeSub}()
    for edge in old_vec
        if edge.prob >= threshold-0.1
            push!(new_vec, edge)
        end
    end
    return new_vec
end   

function choose_5p_less(old_vec::Vector{EdgeSub}, threshold :: Float64)
    new_vec = Vector{EdgeSub}()
    for edge in old_vec
        if edge.prob >= threshold-0.15
            push!(new_vec, edge)
        end
    end
    return new_vec
end   

function random_choice_draw_each(old_vec::Vector{EdgeSub}, threshold:: Float64)
    new_vec = Vector{EdgeSub}()
    for edge in old_vec
        draw = rand()
        if edge.prob >= draw
            push!(new_vec, edge)
        end
    end
    return new_vec
end

function choose_best_cost(old_vec::Vector{EdgeSub}, threshold::Float64)
    new_vec = Vector{EdgeSub}()
    best_edge = old_vec[1]
    best_cost = best_edge.cost
    for edge in old_vec
        if edge.cost < best_cost
            best_edge = edge 
            best_cost = best_edge.cost
        end
    end
    push!(new_vec, best_edge)
    return new_vec
end

function test_choice(new_edge_list::Vector{EdgeSub}, threshold::Float64)
    return new_edge_list
end

function choose_best_prob(old_vec::Vector{EdgeSub}, threshold::Float64)
    new_vec = Vector{EdgeSub}()
    best_edge = old_vec[1]
    best_prob = best_edge.prob
    for edge in old_vec
        if edge.prob > best_prob
            best_edge = edge 
            best_prob = best_edge.time
        end
    end
    push!(new_vec, best_edge)
    return new_vec
end


function choose_best_time(old_vec::Vector{EdgeSub}, threshold::Float64)
    new_vec = Vector{EdgeSub}()
    best_edge = old_vec[1]
    best_time = best_edge.time
    for edge in old_vec
        if edge.cost < best_time
            best_edge = edge 
            best_time = best_edge.time
        end
    end
    push!(new_vec, best_edge)
    return new_vec
end

function no_choice(old_vec::Vector{EdgeSub}, threshold::Float64)
    return old_vec = Vector{EdgeSub}()
end