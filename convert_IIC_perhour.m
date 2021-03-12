clear
feature_hour_folder = 'Z:\Projects\Weilong\Cardiac_arrest_EEG\Features\YNH\';
feature_folder = 'Z:\Projects\Weilong\Cardiac_arrest_EEG\Features_CNN_IIIC\YNH\';
save_folder = 'Z:\Projects\Weilong\Cardiac_arrest_EEG\';
pts_info = readtable('Z:\Projects\Weilong\Cardiac_arrest_EEG\ICARE_CaseData_Final_20181108.csv');

list = dir([feature_folder,'*.mat']);
isfile=~[list.isdir]; 
featurenames={list(isfile).name};
num_file = length(featurenames);
filenames = featurenames;
max_hour = 0;
for i = 1:num_file
    underlineLocations = find(filenames{i} == '_');
    pt_name{i} = filenames{i}(1:underlineLocations(1)-1);
    load ([feature_hour_folder,filenames{i}],'eeg_hour');
    if ~isempty(eeg_hour)
    if max_hour<eeg_hour{end,1}
        max_hour = eeg_hour{end,1};
    end
    end
end

[unique_names, idx ,idx2] = uniquecell(pt_name);
num_pts = length(unique_names);
feature_all = NaN(num_pts,max_hour);
score_all = NaN(num_pts,max_hour);
cpc_scores = zeros(num_pts,1);
features = cell(1,6);
feature_score = cell(1,6);
for ifea = 1:6
    features{ifea} = feature_all;
    feature_score{ifea} = score_all;
end

select_pts = table();

for i = 1:num_pts
    current_pts = unique_names{i};
    indx = ~cellfun(@isempty, strfind(pts_info.sid,current_pts));
    indx_tmp = find(indx==true);
    bestCpcBy6Mo = pts_info.bestCpcBy6Mo(indx_tmp(1));
    cpc_scores(i) = bestCpcBy6Mo;
    unique_names_cpc{i} = [unique_names{i},'(',num2str(bestCpcBy6Mo),')'];
    file_index = find(idx2==i);
    
        for j = 1:length(file_index)

            load([feature_hour_folder,filenames{file_index(j)}],'eeg_hour');
            load([feature_folder,filenames{file_index(j)}],'prediction');
            
            load(['Z:\Projects\Weilong\Cardiac_arrest_EEG\Features\YNH\',filenames{file_index(j)}],'eeg_masks')
            
            col_num_count = 0;
            for k = 1:size(eeg_hour,1)
                if ~isempty(eeg_hour{k,1})
                index = cellfun(@(x)isequal(x,eeg_hour{k,1}),eeg_hour);
                [row,col] = find(index);

                current_mask = eeg_masks(k,col);
%                 current_feature = eeg_feature(k,col);
                weight_clean = zeros(1,length(col));
                feature_raw = zeros(1,length(col));
                end_index = min((col_num_count+length(col))*30,size(prediction,1));
                current_feature = prediction(col_num_count*30+1:end_index,:); % convert 10s to 5min
                col_num_count = col_num_count+length(col);
                
                for ifeature = 1:6
                for m = 1:length(col)
%                     if ~isempty(current_feature{m})
                    indx = ~cellfun(@isempty, strfind(string(current_mask{m}),'normal'));
                    weight_clean(m) = sum(indx)/length(indx);

    %                 feature_raw(m) = mean(current_feature{m}(5:13:end));
                    end_tmp = min(m*30,size(current_feature,1));
                    feature_raw(m) = mean(current_feature((m-1)*30+1:end_tmp,ifeature));

    %                 feature_raw(m) = mean(mean(current_feature{m})); %burst suppression ratio
%                     end
                end
%                 if sum(feature_raw)~=0
                weight_clean_norm = weight_clean/sum(weight_clean);
                feature_clean = sum(weight_clean_norm.*feature_raw);

    %             if max(weight_clean)<0.5
    %                 feature_all(i,eeg_hour{k,1}) = NaN;
    %             else
    %                 feature_all(i,eeg_hour{k,1}) = feature_clean;
    %             end
                features{ifeature}(i,eeg_hour{k,1}) = feature_clean;
                feature_score{ifeature}(i,eeg_hour{k,1}) = max(weight_clean);
%                 end
                end
                end
            end
        end
end
%     features{ifeature} = feature_all;
%     feature_score{ifeature} = score_all;
% end

save([save_folder,'ynh_iic_pattern'],'features','feature_score','cpc_scores','unique_names','unique_names_cpc','-v7.3');
% writetable(select_pts,'ynh_pts.csv');