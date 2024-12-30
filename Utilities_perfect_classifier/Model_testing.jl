

using JuMP, GLPK
model = Model(GLPK.Optimizer)
@variable(model, x[1:3]>=0)
@objective(model, Max, 2*sum(x))
@constraint(model,lol, sum(x) <= 100)
@constraint(model, cons[i = 1:length(x)], x[i] <= 40)

optimize!(model)

JuMP.objective_value(model)


optimize!(model)

new_var = @variable(model, base_name = "x[$(length(x)+1)]", lower_bound >= 0)
push!(x, new_var)
push!(cons, @constraint(model, base_name = "cons[4]",  x[4] <= 10))
JuMP.set_objective_coefficient(model, x[4], 3)


set_normalized_coefficient(lol, new_var, 1)


optimize!(model)

JuMP.objective_value(model)
