% Multiscale LSTM combining fine-grained short-term dynamics (each 6 h time
% block) and coarst long-term dynsmics (all past recordings up to now).
% Wei-Long Zheng, MGH

% short term lstms
clear
close all
load('feature_sequences\all_features_sequences.mat')
save_path = ('multi_scale\');
for num_neuron = 30
    preds_all = {};
    labels_all = {};
    pred_probability_all = {};
    net_model_all = {};
    pts_id_all = {};
    
    for itrial = 1:10
time_step = 6;
time_range = 12:time_step:96;
num_fold = 5;

preds = cell(length(time_range),num_fold);
labels = cell(length(time_range),num_fold);
pred_probability = cell(length(time_range),num_fold);
net_model = cell(length(time_range),num_fold);
for i = length(time_range)-5%1:length(time_range)

    % short sequneces
    bs_x = bs(:,time_range(i)-time_step+1:time_range(i),:);
    bs_x = permute(bs_x,[1,3,2]);
    bs_x = reshape(bs_x,[size(bs_x,1),size(bs_x,2)*size(bs_x,3)]);
    spike_x = spike(:,time_range(i)-time_step+1:time_range(i),:);
    spike_x = permute(spike_x,[1,3,2]);
    spike_x = reshape(spike_x,[size(spike_x,1),size(spike_x,2)*size(spike_x,3)]);
    x = cat(3, bs_x, spike_x);

    for ifea = 1:7
        feature_tmp = features{ifea}(:,time_range(i)-time_step+1:time_range(i),:);
        feature_tmp = permute(feature_tmp,[1,3,2]);
        feature_tmp = reshape(feature_tmp,[size(feature_tmp,1),size(feature_tmp,2)*size(feature_tmp,3)]);
        if ifea==4||ifea==5||ifea==6||ifea==7
            feature_tmp = 10*log10(feature_tmp);
        end
        x = cat(3, x, feature_tmp);
    end
    
    % long sequences
    bs_x = bs(:,time_range(1)-time_step+1:time_range(i),:);
    bs_x = permute(bs_x,[1,3,2]);
    bs_x = reshape(bs_x,[size(bs_x,1),size(bs_x,2)*size(bs_x,3)]);
    spike_x = spike(:,time_range(1)-time_step+1:time_range(i),:);
    spike_x = permute(spike_x,[1,3,2]);
    spike_x = reshape(spike_x,[size(spike_x,1),size(spike_x,2)*size(spike_x,3)]);
    x_long = cat(3, bs_x, spike_x);

    for ifea = 1:7
        feature_tmp = features{ifea}(:,time_range(1)-time_step+1:time_range(i),:);
        feature_tmp = permute(feature_tmp,[1,3,2]);
        feature_tmp = reshape(feature_tmp,[size(feature_tmp,1),size(feature_tmp,2)*size(feature_tmp,3)]);
        if ifea==4||ifea==5||ifea==6||ifea==7
            feature_tmp = 10*log10(feature_tmp);
        end
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
%             index = find(isinf(x_long_tmp(itmp,:)));
%             for idex = 1:length(index)
%                 if ~isinf(x_long_tmp(itmp,index(idex)-1))
%                     x_long_tmp(itmp,index(idex)) = x_long_tmp(itmp,index(idex)-1);
%                 else
%                     x_long_tmp(itmp,index(idex)) = x_long_tmp(itmp,index(idex)+1);
%                 end
%             end
        end
        
        % reshape inputs of long term lstm with the same dimensions
        reduced_dim = 72*2;
        x_long_tmp_short = zeros(9,reduced_dim);
        for itmp = 1:size(x_long_tmp,1)
            if size(x_long_tmp,2)>reduced_dim
                n = fix(size(x_long_tmp,2)/72/2); % for every n points, generate 1 points
                b = arrayfun(@(i) mean(x_long_tmp(itmp,i:i+n-1),2),sort([1:size(x_long_tmp,2)/72:size(x_long_tmp,2),n:size(x_long_tmp,2)/72:size(x_long_tmp,2)])); % the averaged vector
                x_long_tmp_short(itmp,:) = b;
            end
        end
        
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
%     Y = categorical(Y);
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
        train_set = [train_data_short,train_data_long,train_label];
        test_set = [test_data_short,test_data_long,test_label];
        
        pts_id_test = pts_id(start_index:end_index);
        
        ipath_short = [save_path,num2str(i),'h','_train_short_',num2str(round(rand(1)*10e6)),'\'];
        mkdir(ipath_short);
        for itrain = 1:length(train_data_short)
            train_samples = train_data_short{itrain};
            save([ipath_short,num2str(itrain)],'train_samples');
        end
        ipath_long = [save_path,num2str(i),'h','_train_long_',num2str(round(rand(1)*10e6)),'\'];
        mkdir(ipath_long);
        for itrain = 1:length(train_data_long)
            train_samples = train_data_long{itrain};
            save([ipath_long,num2str(itrain)],'train_samples');
        end
        ipath_label = [save_path,num2str(i),'h','_train_label_',num2str(round(rand(1)*10e6)),'\'];
        mkdir(ipath_label);
        for itrain = 1:length(train_label)
            train_labels = train_label{itrain};
            save([ipath_label,num2str(itrain)],'train_labels');
        end
        
        fds_short = fileDatastore(ipath_short,'ReadFcn',@load_variable,'FileExtensions','.mat');
        fds_long = fileDatastore(ipath_long,'ReadFcn',@load_variable,'FileExtensions','.mat');
        fds_label = fileDatastore(ipath_label,'ReadFcn',@load_variable,'FileExtensions','.mat');
        train_datastore = combine(fds_short,fds_long);
        
        %% LSTM
        miniBatchSize = 150;
        maxEpochs = 100;
        layers_short = [ ...
            sequenceInputLayer(9,'Name','InputLayer')
            sequenceFoldingLayer('Name','fold')
            splittingLayer('Splitting-1st','1st')
            bilstmLayer(num_neuron,'OutputMode','sequence','Name','lstm1_short')
            dropoutLayer(0.1,'Name','dropout1_short')
            bilstmLayer(num_neuron,'OutputMode','sequence','Name','lstm2_short')
%             dropoutLayer(0.1)
            bilstmLayer(num_neuron,'OutputMode','sequence','Name','lstm3_short')
%             dropoutLayer(0.1)
            bilstmLayer(num_neuron,'OutputMode','last','Name','lstm4_short')
%             dropoutLayer(0.1)
            fullyConnectedLayer(num_neuron,'Name','fc_short')
            concatenationLayer(1,2,'Name','cat')
%             additionLayer(2,'Name','add')
            fullyConnectedLayer(2,'Name','fc')
            softmaxLayer('Name','softmax_short')
            classificationLayer('Name','classOutput')
            ];
        layers_long = [ ...
%             sequenceInputLayer(9,'Name','input_long')
            splittingLayer('Splitting-2nd','2nd')
            bilstmLayer(num_neuron,'OutputMode','sequence','Name','lstm1_long')
            dropoutLayer(0.1,'Name','dropout1_long')
            bilstmLayer(num_neuron,'OutputMode','sequence','Name','lstm2_long')
%             dropoutLayer(0.1)
            bilstmLayer(num_neuron,'OutputMode','sequence','Name','lstm3_long')
%             dropoutLayer(0.1)
            bilstmLayer(num_neuron,'OutputMode','last','Name','lstm4_long')
%             dropoutLayer(0.1)
            fullyConnectedLayer(num_neuron,'Name','fc_long')
%             softmaxLayer('Name','softmax_long')
            ];
        lgraph = layerGraph(layers_short);
        lgraph = addLayers(lgraph,layers_long);
        lgraph = connectLayers(lgraph,'fc_long','cat/in2');
        layers = connectLayers(lgraph,'InputLayer','Splitting-2nd');
%         lgraph = addLayers(lgraph,sequenceInputLayer(9,'Name','input'));
%         lgraph = connectLayers(lgraph,'input','lstm1_short');
%         lgraph = connectLayers(lgraph,'input','lstm1_long');
        figure,plot(lgraph)

        options = trainingOptions('sgdm', ...%adam sgdm
            'MaxEpochs',maxEpochs, ...
            'MiniBatchSize', miniBatchSize, ...
            'InitialLearnRate', 0.1, ... %0.8
            'ExecutionEnvironment',"cpu",...%'GradientThreshold', 1, ... 
            'Shuffle','never', ... %every-epoch
            'plots','training-progress', ...%training-progress none
            'ValidationData',{test_set(:,1:2),categorical(test_label)},...
            'Verbose',false);%'OutputFcn', @(info)savetrainingplot(info)
        
        % concatenation or addition
        net = trainNetwork(train_datastore,train_label,lgraph,options);
        [pred,probabilities] = classify(net,test_data);
        
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
save(['D:\Research\Cardiac_arrest_EEG\Codes\ComaPrognosticanUsingEEG-master\deep_learning_results\four_layers\','bilstm_four_neurons_',num2str(num_neuron),'_epoch_',num2str(maxEpochs)],'preds','labels','layers','options','pred_probability','net_model');
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
c%     'Verbose',true);
% % ... Training code ...
% % At the end of the script:
% function stop = SaveTrainingPlot(info)
% stop = false;
% if info.State == "done"
%     currentfig = findall(groot,'Type','Figure');
%     savefig(currentfig,'prova.png')
% end
% end
