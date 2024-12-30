using Flux, CUDA, Statistics, ProgressMeter
import MLDataUtils
import MLUtils
using DataFrames,CSV
using ScikitLearn
using BSON: @save,@load
function df_2_dataset_static(filenames)
    to_remove = ["in_path","instance_name","edge_id","iteration","sub_graph_id","lambda_ij","arcs_lambda_i_sub_min","arcs_lambda_i_sub_max","arcs_lambda_i_sub_mean","arcs_lambda_j_sub_min","arcs_lambda_j_sub_max","arcs_lambda_j_sub_mean","pi_ij","times_chosen_for_sub","meanpj","max_pj","min_pj","sink_edge","load_edge","forfeit_edge","dist_less"]
    
    Xs = []
    Ys = []
    name = nothing
    for filename in filenames
        df = CSV.read(filename, DataFrame, ntasks=16)
        for name in names(df)
            if contains(name,"base")
                push!(to_remove,name)
            end
        end

        # drop unused collumns
        df = filter(row->row.load_edge != 1,df)
        df = filter(row->row.sink_edge != 1,df)
        df = filter(row->row.forfeit_edge != 1,df)
        #df = filter(row->row.voy_edge != 1,df)



        # extract target variable
        Y = df[!,"in_path"]
        class_bal = sum(Y)/length(Y)

        select!(df, Not([Symbol(rem) for rem in to_remove]))

        
        println(class_bal)
        println("$filename has $(size(df))")
        push!(Xs,deepcopy(transpose(Matrix(df))))
        push!(Ys,Y)
    end
    if length(filenames) > 1
        X = hcat(Xs...)
        Y = vcat(Ys...)
    else
        Y = Ys[1]
        X = Xs[1]
    end
    class_bal = sum(Y)/length(Y)
    return X,Y,class_bal
end




@load "static_oall_t100_Uea_voytrans_n16_d01.bson" model
sovs = deepcopy(model)
@load "static_oall_t100_trans_16_drop01_new.bson" model
@save "ExperimentModels/sovs.bson" sovs 