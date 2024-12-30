using Flux, CUDA, Statistics, ProgressMeter
using Random
import MLDataUtils
import MLUtils
using DataFrames,CSV
using ScikitLearn
using BSON: @save,@load
using Plots
@sk_import metrics : accuracy_score
@sk_import metrics: recall_score
@sk_import metrics: precision_score
@sk_import model_selection: StratifiedKFold
@sk_import linear_model: LogisticRegression



function df_2_dataset(filenames,edges::Vector{String}, static::Bool, oall::Bool,return_df::Bool)
    Xs = []
    Ys = []
    dfs = []
    Ws = []
    to_remove_after = ["instance_name","edge_id","sub_graph_id","in_path","iteration","in_opt"]
    to_remove = ["load_edge","forfeit_edge","sink_edge","times_chosen_for_sub"]
    use_voy = true
    use_trans = true
    if !("voy_edge" in edges)
        use_voy = false
    end 
    if !("trans_edge" in edges)
        use_trans = false
    end 
    if !(use_voy & use_trans)
        push!(to_remove_after,"trans_edge")
        push!(to_remove_after,"voy_edge")
    end   
    name = nothing
    for filename in filenames
        df = CSV.read(filename, DataFrame, ntasks=16)

        # Filter col names for removal
        for name in names(df)
            if contains(name,"base")
                push!(to_remove,name)
            end
            if contains(name,"max")
                push!(to_remove,name)
            end

            if contains(name,"pi")
                push!(to_remove,name)
            end
            if contains(name,"pj")
                push!(to_remove,name)
            end
            if static
                if contains(name,"lambda") 
                     push!(to_remove,name)
                end
            end
            if oall 
                if contains(name,"dist_less")
                    push!(to_remove,name)  
                end
                if contains(name,"dest")
                    push!(to_remove,name)  
                end
            end
        end
        df = filter(row->row.iteration != 1,df)
        # drop unused collumns
        df = filter(row->row.load_edge != 1,df)
        df = filter(row->row.sink_edge != 1,df)
        df = filter(row->row.forfeit_edge != 1,df)
        if !use_trans
            df = filter(row->row.trans_edge != 1,df)
        end
        if !use_voy
            df = filter(row->row.voy_edge != 1,df)
        end

        # extract target
        Y = df[!,"in_path"]
        W = df[!,"in_opt"]
        W = [(1+w) for w in W]
        class_bal = sum(Y)/length(Y)

        # remove unused collumns
        select!(df, Not([Symbol(rem) for rem in to_remove]))
        push!(dfs,deepcopy(df))
        select!(df, Not([Symbol(rem) for rem in to_remove_after]))
        println("Using cols $(names(df))")
        push!(Xs,deepcopy(transpose(Matrix(df))))
        push!(Ys,Y)
        push!(Ws,W)

    end
    if length(filenames) > 1
        X = hcat(Xs...)
        DF = vcat(dfs...)
        Y = vcat(Ys...)
        W = vcat(Ws...)
    else
        Y = Ys[1]
        X = Xs[1]
        W = Ws[1]
        DF = dfs[1]
    end
    ### add model prediction probabilities

    class_bal = sum(Y)/length(Y)
    if return_df
        return X,Y,W,class_bal,DF
    else
        return X,Y,W,class_bal
    end
end

function prediction_info_edge(y_pred,y_true,x)
    
    # in the features, last four collumns are categorical to indicate edge type
    types = ["voy_edge"]
    
    num_dict = Dict(type => 0 for type in types)
    correct_dict = Dict(type => 0 for type in types)
    for (i , row) in enumerate(eachrow(x))
    type = ""
        if row[end] != 0
            type = "voy_edge"
        end

        num_dict[type]+=1
        if y_pred[i] == y_true[i]
            correct_dict[type]+=1
        end
    end
    for type in types
        println("$type: total: $(num_dict[type]). percent correct $(correct_dict[type]/num_dict[type])")
    end
end
function rec_prec_graph(y_true,y_pred,y_prob,df,filename)
    df[!,"y_pred"] = y_pred
    df[!,"y_true"] = y_true
    df[!,"y_prob"] = y_prob
    thresholds = LinRange(0,0.99,200)
    recs = []
    precs = []
    for threshold in thresholds
        y_pred_new = df[!,"y_prob"] .> threshold
        push!(recs,recall_score(df[!,"y_true"], y_pred_new))
        push!(precs,precision_score(df[!,"y_true"], y_pred_new))
    end
    p1 = plot(thresholds,[recs,precs],label= ["recall" "precision"],xlabel = "threshold", ylabel = "recall/precision")
    savefig(p1, filename)    
    return 
end


function train(train_files,edges, static::Bool)
    ### Training Data ###
    x_train, y_train, w_train, pos_class_fraq_train = df_2_dataset(train_files,edges,static,true,false)
    x_train = Float32.(x_train)
    x_train,w_train, y_train = MLDataUtils.oversample((x_train,w_train,y_train),shuffle = false)
    train_target = Flux.onehotbatch(copy(y_train), [true, false])
    train_data = MLUtils.shuffleobs((x_train,w_train,train_target))
    train_data, val_data = MLDataUtils.stratifiedobs(train_data,p = 0.80) 

    



 
    # define grid search space
    lr = 0.0002
    neurons = 60
    
    input_size = size(train_data[1])[1]
    model = Chain(
        Dense(input_size => neurons, relu),
        Dropout(0.15),
        Dense(neurons=> neurons, relu),
        Dropout(0.15),
        Dense(neurons => 2)) 
    model = model |> gpu

    optim = Flux.setup(Flux.Adam(lr), model)  # will store optimiser momentum, etc.
    losses = []
    epochs = []
    val_loss = []
    val_recall = []
    val_precision = []
    val_accuracy = []
    train_loss = []
    train_recall = []
    train_precision = []
    train_accuracy = []
    batch_size = 2^10

    patience = 6  # Number of epochs to wait for improvement
    min_delta = 0.0001  # Minimum change in validation loss to qualify as improvement
    best_loss = Inf
    wait = 0
    @showprogress for epoch in 1:1000
        loss = 0
        for (x,w, target) in MLUtils.eachobs(train_data, batchsize=batch_size)
            x = x |> gpu
            target = target |> gpu
            w = w |> gpu
            loss, grads = Flux.withgradient(model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x)
                #custom_loss_2(y_hat, target,pos_class_fraq_train)
                Flux.logitcrossentropy(y_hat, target,agg = x-> mean(w .* x))
            end
            Flux.update!(optim, model, grads[1])
        end
        # every 10 epochs get val loss and other metrics
        if (epoch-1) % 2 == 0
            model = model |> cpu
            val_l = []
            val_r = []
            val_p = []
            val_a = []     
            for (x_val,w, target_val) in MLUtils.eachobs(val_data, batchsize=batch_size*8)
                y_pred = model(x_val)  # Prediction
                probs = softmax(y_pred) # normalise to get probabilities
                y_hat = (probs[1,:] .> 0.5)
                y_true = (target_val[1,:] .== 1)
                push!(val_l,Flux.logitcrossentropy(y_pred, target_val,agg = x-> mean(w .* x)))
                push!(val_r,recall_score(y_hat, y_true,zero_division = 0))
                push!(val_p,precision_score(y_hat, y_true,zero_division = 0))
                push!(val_a,accuracy_score(y_hat, y_true))
            end    
            push!(val_loss,mean(val_l))
            push!(val_recall,mean(val_r))
            push!(val_precision,mean(val_p))
            push!(val_accuracy,mean(val_a))
            train_l = []
            train_r = []
            train_p = []
            train_a = [] 
            for (x_train,w,target_train) in MLUtils.eachobs(train_data, batchsize=batch_size*8)
                y_pred = model(x_train) # Prediction
                probs = softmax(y_pred) # normalise to get probabilities
                y_hat = (probs[1,:] .> 0.5)
                y_true = (target_train[1,:] .== 1)
                push!(train_l,Flux.logitcrossentropy(y_pred, target_train,agg = x-> mean(w .* x)))
                push!(train_r,recall_score(y_hat, y_true,zero_division = 0))
                push!(train_p,precision_score(y_hat, y_true,zero_division = 0))
                push!(train_a,accuracy_score(y_hat, y_true))
            end 
            push!(train_loss,mean(train_l))
            push!(train_recall,mean(train_r))
            push!(train_precision,mean(train_p))
            push!(train_accuracy,mean(train_a))

            model = model |> gpu
            println("Train data: loss $(train_loss[end]), recall:$(train_recall[end]), precision: $(train_precision[end]), accuracy: $(train_accuracy[end])")
            println("Val data: loss $(val_loss[end]), recall:$(val_recall[end]), precision: $(val_precision[end]), accuracy: $(val_accuracy[end])")

            if val_l[end] < best_loss - min_delta
                best_loss = val_l[end]
                wait = 0  # Reset patience counter if validation loss improves
            else
                wait += 1  # Increment patience counter
                println("waiting step number $wait")
            end
            if wait >= patience
                println("Early stopping at epoch $epoch")
                break
            end
        end

    push!(losses,loss)
    #println("Current loss at epoch $epoch is $loss")
    push!(epochs,epoch)
    end

    model = model|>cpu

    unseen = split(test_files[1],"/")[2]



    @save "NN_models_inopt/"*unseen*".bson" model
end
data_folder = "Data_inopt/"
all_csv = [data_folder * i for i in readdir(data_folder)]

static_csv = []
dynamic_csv = []

for csv in all_csv 
    if contains(csv, "dynamic")
        push!(dynamic_csv,csv)
    else
        push!(static_csv,csv)
    end
end 
edges = ["voy_edge","trans_edge"]
train_files = []
# make static models
for (i_test,test_instance) in enumerate(static_csv) 
        train_files = static_csv[1:end .!= i_test]   
        println("test $test_instance")
        train(train_files,[test_instance],edges,true)
end 


# make dynamic models
edges = ["voy_edge","trans_edge"]
train_files = []
# make static models
for (i_test,test_instance) in enumerate(dynamic_csv) 
    train_files = dynamic_csv[1:end .!= i_test]   
    println("test $test_instance")
    train(train_files,[test_instance],edges,false)
end 


##### random forrest models #####
using MLJ
using ScikitLearn
using MLJScikitLearnInterface
using PyCall
using CategoricalArrays





model_folder = "RF_models_inpath/"
data_folder = "Data_inpath/"
all_csv = [data_folder * i for i in readdir(data_folder)]
params = (20, 4, 200, 4)
function train_rf(x,y,w,params)
    y = CategoricalArray(y)

    maxdepth,minsamplessplit,nestimators,minsamplesleaf = params
    clf = RandomForestClassifier(n_estimators = nestimators,
                            min_samples_leaf = minsamplesleaf,
                            min_samples_split = minsamplessplit,
                            max_depth = maxdepth,
                            n_jobs = 1)
    mach = machine(clf, x', y,w)
    MLJ.fit!(mach)                                            
    # test_acc = score(rf,x_test',y_test,sample_weight = w_test)

    return mach
end

static_csv = []
dynamic_csv = []

for csv in all_csv 
    if contains(csv, "dynamic")
        push!(dynamic_csv,csv)
    else
        push!(static_csv,csv)
    end
end 
edges = ["voy_edge","trans_edge"]
train_files = []

# make static models
static = true
for (i_test,test_instance) in enumerate(static_csv) 
    train_files = static_csv[1:end .!= i_test]   
    println("test $test_instance")
    x, y, w, pos_class_fraq_train = df_2_dataset([train_files],edges,static,true,false)
    x = Float32.(x)
    x,w,y = MLDataUtils.oversample((x,w,y),shuffle = true)
    mach = train_rf(x,y,w,params)
    test_instance = replace(test_instance,data_folder => "")
    MLJ.save(model_folder*test_instance*"_RF.jlso", mach)
end 

# make dynamic models
edges = ["voy_edge","trans_edge"]
train_files = []
# make static models
static = false
for (i_test,test_instance) in enumerate(dynamic_csv) 
    train_files = dynamic_csv[1:end .!= i_test]   
    println("test $test_instance")
    x, y, w, pos_class_fraq_train = df_2_dataset([train_files],edges,static,true,false)
    x = Float32.(x)
    x,w,y = MLDataUtils.oversample((x,w,y),shuffle = true)
    mach = train_rf(x,y,w,params)
    test_instance = replace(test_instance,data_folder => "")
    MLJ.save(model_folder*test_instance*"_RF.jlso", mach)
end 


function visual_eval(y_true,y_pred,y_prob,df)
    #merge y's into df
    df[!,"y_pred"] = y_pred
    df[!,"y_true"] = y_true
    df[!,"y_prob"] = y_prob

    ##initialize containers
    #fn_res_dict = Dict()
    #tp_res_dict = Dict()
    #tn_res_dict = Dict()
    #fp_res_dict = Dict()

    ## containers for subgraph 
    #S_edges = []
    #S_edges_pos = []
    #S_recall = []
    #S_accuracy = []
    #S_precision = []
    #S_residuals = []

    ## containers for Instances
    #I_name = []
    #I_edges = []
    #I_recall = []
    #I_accuracy = []
    #I_precision = []
    #I_residuals = []

    ## based on edge type:
    #for instance_name in Set(df[!,"instance_name"])
    #    push!(I_name,instance_name)
    #    df_i = filter(row->row.instance_name == instance_name,df)
    #    # instance metrics
    #    push!(I_recall, recall_score(df_i[!,"y_true"],df_i[!,"y_pred"]))
    #    push!(I_precision, precision_score(df_i[!,"y_true"],df_i[!,"y_pred"]))
    #    push!(I_accuracy,accuracy_score(df_i[!,"y_true"],df_i[!,"y_pred"]))
    #    push!(I_edges, size(df_i)[1])
    #    push!(I_residuals, df_i[!,"y_prob"] .- 0.5 )
    #    # FN residuals
    #    df_i_fn = filter(row->row.y_true == 1 && row.y_pred == 0,df_i)
    #    fn_res_dict[instance_name] = df_i_fn[!,"y_prob"] .- 0.5 
    #    # TP residuals
    #    df_i_tp = filter(row->row.y_true == 1 && row.y_pred == 1,df_i)
    #    tp_res_dict[instance_name] = df_i_tp[!,"y_prob"] .- 0.5 
    #    # TN residuals
    #    df_i_tn = filter(row->row.y_true == 0 && row.y_pred == 0,df_i)
    #    tn_res_dict[instance_name] =df_i_tn[!,"y_prob"] .- 0.5 
    #    # FP residuals
    #    df_i_fp = filter(row->row.y_true == 0 && row.y_pred == 1,df_i)
    #    fp_res_dict[instance_name] = df_i_fp[!,"y_prob"] .- 0.5 

    #    for subgraph_id in Set(df_i[!,"sub_graph_id"])
    #        # subgraph metrics
    #        df_s = filter(row->row.sub_graph_id == subgraph_id,df_i)
    #        println(size(df_s))
    #        push!(S_recall, recall_score(df_s[!,"y_true"],df_s[!,"y_pred"]))
    #        push!(S_precision, precision_score(df_s[!,"y_true"],df_s[!,"y_pred"]))
    #        push!(S_accuracy, accuracy_score(df_s[!,"y_true"],df_s[!,"y_pred"]))
    #        push!(S_edges_pos,sum(df_s[!,"y_true"]))
    #        push!(S_edges,size(df_s)[1])
    #    end
    #end

    #### PLOTS ###
    #
    ## residuals
    #for I in I_name
    #    p1 = bar(sort(fn_res_dict[I]),title ="FN")
    #    p2 = bar(sort(tn_res_dict[I]),title ="TN")
    #    p3 = bar(sort(tp_res_dict[I]),title ="TP")
    #    p4 = bar(sort(fp_res_dict[I]),title ="FP")
    #    barplot = plot(p1,p2,p3,p4,layoyt = (2,2),plot_title= I)
    #    display(barplot)
    #end
    #
    #### metrics vs # number of edges
    #fig_e_r = scatter(S_edges,S_recall,ma = 0.6,xlabel = "Number of edges",ylabel = "Recall",title = "Recall/#edges for all subgraphs")
    #fig_e_p = scatter(S_edges,S_precision,ma = 0.6,xlabel = "Number of edges",ylabel = "Precision",title = "precision/#edges for all subgraphs")
    #fig_e_a = scatter(S_edges,S_accuracy,ma = 0.6,xlabel = "Number of edges",ylabel = "Accuracy",title = "accuracy/#edges for all subgraphs")
    #
    #
    #
    #### metrics vs number of positive labels
    #fig_pe_r = scatter(S_edges_pos./S_edges,S_recall,ma = 0.6,xlabel = "Number of pos edges",ylabel = "Recall",title = "Recall/#pos edges for all subgraphs")
    #fig_pe_p = scatter(S_edges_pos./S_edges,S_precision,ma = 0.6,xlabel = "Number of pos  edges",ylabel = "Precision",title = "precision/#pos edges for all subgraphs")
    #fig_pe_a = scatter(S_edges_pos./S_edges,S_accuracy,ma = 0.6,xlabel = "Number of pos edges",ylabel = "Accuracy",title = "accuracy/#pos edges for all subgraphs")
    #
    #display(fig_e_r)
    #display(fig_e_p)
    #display(fig_e_a)
    #display(fig_pe_r)
    #display(fig_pe_p)
    #display(fig_pe_a)
    ### recall/precision as function of threshold
    thresholds = LinRange(0,0.99,200)
    recs = []
    precs = []
    for threshold in thresholds
        y_pred_new = df[!,"y_prob"] .> threshold
        push!(recs,recall_score(df[!,"y_true"], y_pred_new))
        push!(precs,precision_score(df[!,"y_true"], y_pred_new))
    end
    p1 = plot(thresholds,[recs,precs],label= ["recall" "precision"],xlabel = "threshold", ylabel = "recall/precision")
    display(p1)


    return 
end

######### EVALUATION ##########
# static
model_folder = "EvenNewerModels2/"
ps = []
for (i,file) in enumerate(static_csv)
    x_train,y_train,pos_class_fraq_train = df_2_dataset([file],edges,true,true,false)
    model = nothing
    for model_file in readdir(model_folder)
        if contains(file,split(model_file,".")[1])
            @load model_folder * model_file model 
        end   
    end
    y_pred = model(x_train) # Prediction
    probs = softmax(y_pred) # normalise to get probabilities
    y_true = y_train
    y_prob = probs[1,:]

    ### make plot
    thresholds = LinRange(0,0.99,200)
    recs = []
    precs = []
    for threshold in thresholds
        y_pred_new = y_prob .> threshold
        push!(recs,recall_score(y_true, y_pred_new))
        push!(precs,precision_score(y_true, y_pred_new))
    end
    p1 = plot(thresholds,[recs,precs],label= ["recall" "precision"],xlabel = "threshold", ylabel = "recall/precision",title = file)
    push!(ps,p1)
end
plot(ps...,layout=(1,7))


x_train,y_train,pos_class_fraq_train = df_2_dataset([train_file],edges,false,true,false)
x_train = vcat(x_train,transpose(train_probs[1,:]))
x_train = Float32.(x_train)

x_test,y_test,pos_class_fraq_test= df_2_dataset([test_file],edges,false,true,false)
x_test= vcat(x_test,transpose(test_probs[1,:]))
x_test = Float32.(x_test)


@load "ExperimentModels/dynamic_oall_t100_Uws_voytrans_n20_d01_statprobs.bson" model

### static prec rec plot
@load "ExperimentModels/dynamic_oall_t100_Uws_voytrans_n20_d01_statprobs.bson" model

thresholds = LinRange(0,0.99,200)
recs = []
precs = []
for threshold in thresholds
    y_pred_new = df[!,"y_prob"] .> threshold
    push!(recs,recall_score(df[!,"y_true"], y_pred_new))
    push!(precs,precision_score(df[!,"y_true"], y_pred_new))
end
p1 = plot(thresholds,[recs,precs],label= ["recall" "precision"],xlabel = "threshold", ylabel = "recall/precision")
display(p1)



y_pred = model(x_train) # Prediction
probs = softmax(y_pred) # normalise to get probabilities
y_hat = (probs[1,:] .> 0.5)
y_true = y_train
train_recall = recall_score(y_true, y_hat)
#prediction_info_edge(y_hat,y_true,transpose(x_train))
train_precision = precision_score(y_true, y_hat)
train_accuracy = accuracy_score(y_true, y_hat)
println("$pos_class_fraq_train% positive class balance on test set")
println("Train precision $train_precision")
println("Train recall $train_recall")
println("Train accuracy $train_accuracy")

visual_eval(y_true,y_hat,probs[1,:],df_train)



y_pred = model(x_test) # Prediction
probs = softmax(y_pred) # normalise to get probabilities

y_hat = (probs[1,:] .> 0.5)
y_true = y_test
test_recall = recall_score(y_true, y_hat)
test_precision = precision_score(y_true, y_hat)
test_accuracy = accuracy_score(y_true, y_hat)
println("Test precision $test_precision")
println("Test recall $test_recall")
println("Test accuracy $test_accuracy")
visual_eval(y_true,y_hat,probs[1,:],df_test)

#@save "static_oall_t100_trans.bson" model



thresholds = LinRange(0,0.99,200)
recs = []
precs = []
for threshold in thresholds
    y_pred_new = probs[1,:] .> threshold
    push!(recs,recall_score(y_true, y_pred_new))
    push!(precs,precision_score(y_true, y_pred_new))
end
p1 = plot(thresholds,[recs,precs],label= ["recall" "precision"],xlabel = "threshold", ylabel = "recall/precision")
display(p1)











a = ["cost_of_edge", "time_of_edge", "cap_of_edge", "arc_leaving_i_sub", "arc_leaving_j_sub", "arc_entering_i_sub", "arc_entering_j_sub", "arcs_cost_i_sub_min", "arcs_cost_i_sub_max", "arcs_cost_i_sub_mean", "arcs_time_i_sub_min", "arcs_time_i_sub_max", "arcs_time_i_sub_mean", "arcs_cap_i_sub_min", "arcs_cap_i_sub_max", "arcs_cap_i_sub_mean", "arcs_ent_cost_i_sub_min", "arcs_ent_cost_i_sub_max", "arcs_ent_cost_i_sub_mean", "arcs_ent_time_i_sub_min", "arcs_ent_time_i_sub_max", "arcs_ent_time_i_sub_mean", "arcs_ent_cap_i_sub_min", "arcs_ent_cap_i_sub_max", "arcs_ent_cap_i_sub_mean", "arcs_cost_j_sub_min", "arcs_cost_j_sub_max", "arcs_cost_j_sub_mean", "arcs_time_j_sub_min", "arcs_time_j_sub_max", "arcs_time_j_sub_mean", "arcs_cap_j_sub_min", "arcs_cap_j_sub_max", "arcs_cap_j_sub_mean", "arcs_ent_cost_j_sub_min", "arcs_ent_cost_j_sub_max", "arcs_ent_cost_j_sub_mean", "arcs_ent_time_j_sub_min", "arcs_ent_time_j_sub_max", "arcs_ent_time_j_sub_mean", "arcs_ent_cap_j_sub_min", "arcs_ent_cap_j_sub_max", "arcs_ent_cap_j_sub_mean", "voy_edge", "trans_edge"]















### compare with RF and lr
x_train,y_train,pos_class_fraq_train,df_train = df_2_dataset(["train_oall_BIG_t100_df.csv"],["trans_edge"],true,true,true)
x_train = Float32.(x_train)
x_test,y_test,pos_class_fraq_test,df_test= df_2_dataset(["test_oall_BIG_t100_df.csv"],["trans_edge"],true,true,true)
x_test = Float32.(x_test)
#x_train,w,y_train = train_data
#x_test,w_s,y_test = test_data
#y_train = y_train[1,:].==1
y_true = y_test
x_train = transpose(x_train)
x_test = transpose(x_test)

log_reg = ScikitLearn.fit!(LogisticRegression(class_weight = "balanced",max_iter = 2000), x_train, y_train);
y_hat = ScikitLearn.predict(log_reg,x_test);
test_recall = recall_score(y_true, y_hat);
test_precision = precision_score(y_true, y_hat);
test_accuracy = accuracy_score(y_true, y_hat);
println("Logistic regression")
println("Test precision $test_precision")
println("Test recall $test_recall")
println("Test accuracy $test_accuracy")
prediction_info_edge(y_hat,y_true,x_test)

rf = ScikitLearn.fit!(RandomForestClassifier(class_weight = "balanced"), x_train, y_train);
y_hat = ScikitLearn.predict(rf,x_test);
test_recall = recall_score(y_true, y_hat);
test_precision = precision_score(y_true, y_hat);
test_accuracy = accuracy_score(y_true, y_hat);
println("Random Forrest")
println("Test precision $test_precision")
println("Test recall $test_recall")
println("Test accuracy $test_accuracy")
prediction_info_edge(y_hat,y_true,x_test)

function prediction_info_edge(y_pred,y_true,x)
    # in the features, last four collumns are categorical to indicate edge type
    types = ["voy_edge","trans_edge"]
    
    num_dict = Dict(type => 0 for type in types)
    correct_dict = Dict(type => 0 for type in types)
    for (i , row) in enumerate(eachrow(x))
    type = ""
        if row[end] != 0
            type = "trans_edge"
        elseif row[end-1] != 0
            type = "voy_edge"
        end

        num_dict[type]+=1
        if y_pred[i] == y_true[i]
            correct_dict[type]+=1
        end
    end
    for type in types
        println("$type: total: $(num_dict[type]). percent correct $(correct_dict[type]/num_dict[type])")
    end
end
function prediction_info_edge(y_pred,y_true,x)
    # in the features, last four collumns are categorical to indicate edge type
    types = ["voy_edge","trans_edge"]
    num_features = size(x)[1]
    row_inds = [numfeatures-1,num_features]
    for (i,row_ind) in enumerate(row_inds)
        y_pred_type = y_pred[x[row_ind,:] .==1]
        y_true_type = y_true[x[row_ind,:] .==1]
        println("$(types[i]) recall: $(recall_score(y_pred_type, y_true_type,zero_division = 0))")
        println("$(types[i]) recall: $(precision_score(y_pred_type, y_true_type,zero_division = 0))")
        println("$(types[i]) recall: $(accuracy_score(y_pred_type, y_true_type))")
    end
end

prediction_info_edge(y_hat,y_true,x_test)

df = CSV.read("train_df.csv", DataFrame, ntasks=16)
df = filter(row->row.load_edge != 1,df)
df = filter(row->row.sink_edge != 1,df)