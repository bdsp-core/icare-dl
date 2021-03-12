% Multiscale LSTM combining fine-grained short-term dynamics (each 6 h time
% block) and coarst long-term dynsmics (all past recordings up to now).
% Wei-Long Zheng, MGH

% short term lstms
clear
close all
load('Z:\Projects\Weilong\Cardiac_arrest_EEG\cnn_all_features_sequences.mat')

%%
for num_neuron = 30:10:70%40:10:70%80:10:100%80:10:100% 30:10:70
    preds_all = {};
    labels_all = {};
    pred_probability_all = {};
    net_model_all = {};
    pts_id_all = {};
    
    for itrial = 1%:10
time_step = 6;
time_range = 12:time_step:96;
num_fold = 5;

preds = cell(length(time_range),num_fold);
labels = cell(length(time_range),num_fold);
pred_probability = cell(length(time_range),num_fold);
net_model = cell(length(time_range),num_fold);
for i = 1:length(time_range)

    % short sequneces
    x = [];

    for ifea = 1:1024
        feature_tmp = features{ifea}(:,time_range(i)-time_step+1:time_range(i),:);
        feature_tmp = permute(feature_tmp,[1,3,2]);
        feature_tmp = reshape(feature_tmp,[size(feature_tmp,1),size(feature_tmp,2)*size(feature_tmp,3)]);
        x = cat(3, x, feature_tmp);
    end
    
    % long sequences
    x_long = [];

    for ifea = 1:1024
        feature_tmp = features{ifea}(:,time_range(1)-time_step+1:time_range(i),:);
        feature_tmp = permute(feature_tmp,[1,3,2]);
        feature_tmp = reshape(feature_tmp,[size(feature_tmp,1),size(feature_tmp,2)*size(feature_tmp,3)]);
        x_long = cat(3, x_long, feature_tmp);
    end
    
    cpc_scores_binary = cpc_scores;
    pos_index = find(cpc_scores<3);
    neg_index = find(cpc_scores>=3);
    cpc_scores_binary(pos_index) = 1;
    cpc_scores_binary(neg_index) = 0;
    
    X = {};
    X_long = {};
    Y = {};
    pts_id = {};
    for ipts = 1:size(x,1)
        x_tmp = squeeze(x(ipts,:,:));
        x_tmp = x_tmp';
        x_long_tmp = squeeze(x_long(ipts,:,:));
        x_long_tmp = x_long_tmp';
        
        for itmp = 1:size(x_tmp,1)
            x_tmp(itmp,isinf(x_tmp(itmp,:))) = nan;
            x_long_tmp(itmp,isinf(x_long_tmp(itmp,:))) = nan;
            x_tmp(itmp,isnan(x_tmp(itmp,:))) = nanmean(x_tmp(itmp,:));
            temp = x_long_tmp(itmp,end-time_step*12+1:end);
            temp(1,isnan(temp(1,:))) = nanmean(temp(1,:));
            x_long_tmp(itmp,end-time_step*12+1:end) = temp;
            nan_index = find(isnan(x_long_tmp(itmp,:)));
            for jnan = length(nan_index):-1:1
                if nan_index(jnan)+time_step*12<=size(x_long_tmp,2)
                    x_long_tmp(itmp,nan_index(jnan)) = nanmean(x_long_tmp(itmp,nan_index(jnan)+1:nan_index(jnan)+time_step*12));
                end
            end
        end
        
        % reshape inputs of long term lstm with the same dimensions
%         ihour = 2; 
%         reduced_dim = 72*ihour;
%         x_long_tmp_short = [];
%         for itmp = 1:size(x_long_tmp,1)
%             if size(x_long_tmp,2)>reduced_dim
%                 n = fix(size(x_long_tmp,2)/reduced_dim); % for every n points, generate 1 points
%                 n_index = [];
%                 for i_n = 1:ihour
%                     n_index = [n_index,(i_n-1)*n+1:size(x_long_tmp,2)/72:size(x_long_tmp,2)];
%                 end
%                 n_index = sort(n_index);
%                 b = arrayfun(@(i) mean(x_long_tmp(itmp,i:i+n-1),2),n_index); % the averaged vector
% %                 b = arrayfun(@(i) mean(x_long_tmp(itmp,i:i+n-1),2),sort([1:size(x_long_tmp,2)/72:size(x_long_tmp,2),n:size(x_long_tmp,2)/72:size(x_long_tmp,2)])); % the averaged vector
%                 x_long_tmp_short(itmp,:) = b;
%             else
%                 x_long_tmp_short(itmp,:) = x_long_tmp(itmp,:);
%             end
%         end
        x_long_tmp_short = x_long_tmp;
        
        if ~isnan(sum(sum(x_tmp)))&&~isinf(sum(sum(x_tmp)))
            X = [X; x_tmp];
            X_long = [X_long; x_long_tmp_short];
            Y = [Y; num2str(cpc_scores_binary(ipts))];
            pts_id = [pts_id; unique_names{ipts}];
        end
    end
    
    
    
    
    XV = [X{:}];
    mu = mean(XV,2);
    sg = std(XV,[],2);
    X = cellfun(@(x)(x-mu)./sg,X,'UniformOutput',false);
    XV = [X_long{:}];
    mu = mean(XV,2);
    sg = std(XV,[],2);
    X_long = cellfun(@(x)(x-mu)./sg,X_long,'UniformOutput',false);

    num_pts = length(Y);
    num_test = round(num_pts/num_fold);
    idx = randperm(num_pts);
    X = X(idx);
    X_long = X_long(idx);
    Y = Y(idx);
    pts_id = pts_id(idx);
    Y = categorical(Y);
    for ifold = 1:num_fold
        if ifold~=num_fold
            start_index = (ifold-1)*num_test+1;
            end_index = ifold*num_test;
        else
            start_index = (ifold-1)*num_test+1;
            end_index = num_pts;
        end
        train_index = setdiff(1:num_pts,start_index:end_index);
        
        test_data_short = X(start_index:end_index);
        test_label = Y(start_index:end_index);
        train_data_short = X(train_index);
        train_label = Y(train_index);
        
        test_data_long = X_long(start_index:end_index);
        train_data_long = X_long(train_index);
        
        pts_id_test = pts_id(start_index:end_index);

        %% LSTM
        miniBatchSize = 250;
        maxEpochs = 15;%50;
        layers = [ ...
            sequenceInputLayer(1024)
            bilstmLayer(num_neuron,'OutputMode','sequence')
            dropoutLayer(0.1)
%             bilstmLayer(num_neuron,'OutputMode','sequence')
%             dropoutLayer(0.1)
%             bilstmLayer(num_neuron,'OutputMode','sequence')
%             dropoutLayer(0.1)
            bilstmLayer(num_neuron,'OutputMode','last')
%             dropoutLayer(0.1)
            fullyConnectedLayer(2)
            softmaxLayer
            classificationLayer
            ];
        options = trainingOptions('sgdm', ...%adam sgdm
            'MaxEpochs',maxEpochs, ...
            'MiniBatchSize', miniBatchSize, ...
            'InitialLearnRate', 0.05, ... %0.1
            'ExecutionEnvironment','cpu', ... % Turn on automatic parallel support. 'ExecutionEnvironment',"gpu",...%cpu 'GradientThreshold', 1, ...
            'Shuffle','every-epoch', ...
            'plots','none', ...%training-progress none 
            'ValidationFrequency',5,...
            'ValidationData',{test_data_long,test_label},...
            'Verbose',false);%'OutputFcn', @(info)savetrainingplot(info)
        
        net = trainNetwork(train_data_long,train_label,layers,options);
        [pred,probabilities] = classify(net,test_data_long);
        
        preds{i,ifold} = pred;
        labels{i,ifold} = test_label;
        pred_probability{i,ifold} = probabilities;
        net_model{i,ifold} = net;
        pts_id_fold{i,ifold} = pts_id_test;
    end
end
        preds_all{itrial} = preds;
        labels_all{itrial} = labels;
        pred_probability_all{itrial} = pred_probability;
        net_model_all{itrial} = net_model;
        pts_id_all{itrial} = pts_id_fold;
    end
save(['D:\Research\Cardiac_arrest_EEG\Codes\ComaPrognosticanUsingEEG-master\multiscale-lstm\','super_long_bilstm_two_neurons_',num2str(num_neuron),'_epoch_',num2str(maxEpochs)],'preds_all','labels_all','pts_id_all','layers','options','pred_probability_all','net_model_all');
end

%[updatedNet,YPred] = predictAndUpdateState(recNet,sequences)

% function stop=savetrainingplot(info)
% stop=false;  %prevents this function from ending trainNetwork prematurely
% if info.State=='done'   %check if all iterations have completed
% % if true
%       saveas(gca,'training_process.png')  % save figure as .png, you can change this
% 
% end
% end

% options = trainingOptions('sgdm',...
%     'InitialLearnRate',0.003,...
%     'Plots','training-progress', ...
%     'ValidationData',garVal,...
%     'ValidationFrequency',40,...
%     'MaxEpochs',1,...
%     'LearnRateSchedule', 'piecewise',...
%     'LearnRateDropPeriod',3,...
%     'Shuffle','every-epoch',...
%     'ValidationPatience',5,...
%     'OutputFcn',@(info)SaveTrainingPlot(info),...
%     'Verbose',true);
% % ... Training code ...
% % At the end of the script:
% function stop = SaveTrainingPlot(info)
% stop = false;
% if info.State == "done"
%     currentfig = findall(groot,'Type','Figure');
%     savefig(currentfig,'prova.png')
% end
% end