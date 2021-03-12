clear
pts_info = readtable('Z:\Projects\Weilong\Cardiac_arrest_EEG\ICARE_CaseData_Final_20181108.csv');

neuron_num_short = 30:10:100;
neuron_num_long = 30:10:40;
load(['D:\Research\Cardiac_arrest_EEG\Codes\ComaPrognosticanUsingEEG-master\multiscale-lstm\long_limitSeq_8_bilstm_two_neurons_',num2str(neuron_num_long(2)),'_epoch_30.mat']);
pred_probability_long = pred_probability_all{1};
pts_id_long = pts_id_all{1};
load(['D:\Research\Cardiac_arrest_EEG\Codes\ComaPrognosticanUsingEEG-master\multiscale-lstm\short_bilstm_two_neurons_',num2str(neuron_num_short(3)),'_epoch_15.mat']);
labels_short = labels_all{1};
pred_probability_short = pred_probability_all{1};
pts_id_short = pts_id_all{1};


for itrial = 1%:10
    time_step = 6;
    time_range = 12:time_step:96;
    num_fold = 5;

    preds_clinical = cell(length(time_range),num_fold);
    labels_clinical = cell(length(time_range),num_fold);
    pred_probability_clinical = cell(length(time_range),num_fold);

    for i = 1:length(time_range)
        pts_id_short_tmp = cat(1,pts_id_short{i,:});
        pts_id_long_tmp = cat(1,pts_id_short{i,:});
        prob_long_tmp = cat(1,pred_probability_long{i,:});
        prob_short_tmp = cat(1,pred_probability_short{i,:});
        label_tmp = cat(1,labels_short{i,:});

        X = [];
        Y = [];
        pts_id = {};
        for ipts = 1:length(pts_id_short_tmp)
            pts = pts_id_short_tmp{ipts};
            X_tmp = [];
            indx = ~cellfun(@isempty, strfind(pts_info.sid,pts));
            indx_tmp = find(indx==true);
            pts_sex = pts_info.sex(indx_tmp(1));
            pts_age = pts_info.age(indx_tmp(1));
            pts_vfib = pts_info.vfib(indx_tmp(1));
            if ~isnan(pts_age)&&~isempty(pts_sex{:})&&~isnan(pts_vfib)

                indx = ~cellfun(@isempty, strfind(pts_id_short_tmp,pts));
                indx = find(indx==1);
                indx = indx(1);
                prob_short = prob_short_tmp(indx,1);
                label_current = label_tmp(indx);

                indx = ~cellfun(@isempty, strfind(pts_id_long_tmp,pts));
                indx = find(indx==1);
                indx = indx(1);
                prob_long = prob_long_tmp(indx,1);
                
                p = (prob_short+prob_long)/2;
                X_tmp = [log(p/(1-p)), pts_age/100, double(pts_sex{:}=='M'), pts_vfib];
                
                X = [X; X_tmp];
                Y = [Y; label_current];
                pts_id = [pts_id; pts];
            end

        end
        
        num_pts = length(pts_id);
        num_test = round(num_pts/num_fold);
        idx = randperm(num_pts);
        X = double(X(idx,:));
        Y = double(Y(idx));
        pts_id = pts_id(idx);
        
        pos_ind = find(Y==2);
        neg_ind = find(Y==1);
        Y(pos_ind) = 1;
        Y(neg_ind) = 0;
        
        for ifold = 1:num_fold
            if ifold~=num_fold
                start_index = (ifold-1)*num_test+1;
                end_index = ifold*num_test;
            else
                start_index = (ifold-1)*num_test+1;
                end_index = num_pts;
            end
            train_index = setdiff(1:num_pts,start_index:end_index);
            
            test_data = X(start_index:end_index,:);
            test_label = Y(start_index:end_index);
            train_data = X(train_index,:);
            train_label = Y(train_index);
            pts_id_test = pts_id(start_index:end_index);
            
                    % logistic regression
%             [B,dev,stats] = glmfit(train_data,train_label,'normal','logit');
%             probabilities = glmval(B,test_data,'logit');
%             pred = zeros(length(test_label),1);
%             pred(probabilities(:,2)>0.5)=1;
            
            % random forest
            B = TreeBagger(60,train_data,train_label,'Method','Classification','OOBVarImp','On');
%             weight_tmp=B.OOBPermutedVarDeltaError;
            [pred, probabilities] = predict(B, test_data);
            
            preds{i,ifold} = pred;
            labels{i,ifold} = test_label;
            pred_probability{i,ifold} = probabilities;
            net_model{i,ifold} = B;
            pts_id_fold{i,ifold} = pts_id_test;
        end
    end
    
    preds_all{itrial} = preds;
    labels_all{itrial} = labels;
    pred_probability_all{itrial} = pred_probability;
    net_model_all{itrial} = net_model;
    pts_id_all{itrial} = pts_id_fold;
end
save(['D:\Research\Cardiac_arrest_EEG\Codes\ComaPrognosticanUsingEEG-master\multiscale-lstm\','RF_lstmProb_clinical'],'preds_all','labels_all','pts_id_all','pred_probability_all','net_model_all');
