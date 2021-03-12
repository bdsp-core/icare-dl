%% plot performance curves of individual cpc groups
clear
time_step = 6;
time_range = 12:time_step:96;
neuron_num_short = 30:10:100;
neuron_num_long = 30:10:40;
load(['RF_clinical_regardless_time']);
prob_pts_clinical = prob_pts;
a = mean(prob_pts_clinical');
prob_pts_clinical = repmat(a',1,length(time_range));
load(['\long_limitSeq_8_bilstm_two_neurons_',num2str(neuron_num_long(2)),'_epoch_30.mat']);
prob_pts_long = prob_pts;
load(['\short_bilstm_two_neurons_',num2str(neuron_num_short(3)),'_epoch_15.mat']);
prob_pts_short = prob_pts;

prob_pts = (prob_pts_clinical+prob_pts_long+prob_pts_short)/3;

pred_prob_all = nan(size(prob_pts));
for i = 1:size(pred_prob_all,1)
for j = 1:size(pred_prob_all,2)
    if ~isnan(prob_pts(i,j))
        pred_prob_all(i,j) = nanmean(prob_pts(i,1:j));
    end
end
end
labels = nanmean(label_pts');

num_pts = length(labels);
index_tmp = find(labels==2);
feature_good = pred_prob_all(labels==2,:);
feature_mean = nanmean(feature_good');
[a,index_good] = sort(feature_mean);
index_good = index_tmp(index_good);
feature_good = pred_prob_all(index_good,:);
unique_names_good = unique_names(index_good);

index_tmp = find(labels==1);
feature_bad = pred_prob_all(labels==1,:);
feature_mean = nanmean(feature_bad');
[a,index_bad]=sort(feature_mean);
index_bad = index_tmp(index_bad);
feature_bad = pred_prob_all(index_bad,:);
unique_names_bad = unique_names(index_bad);

unique_names1 = [unique_names_good;unique_names_bad];
feature_all1 = [feature_good;feature_bad];

figure,
% mn = prctile(feature_all1(:),0); mx = prctile(feature_all1(:),100);% 90
imagesc(feature_all1(:,1:11));
colormap(flipud(cold));
% set(gca, 'YTick', 371, 'YTickLabel', 'poor')
xlabel('Hours After Cardiac Arrest');
title('Outcome Prediction Probability over Time');
colorbar;
set(gca, 'Xtick',1:11,'Xticklabel',num2cell(time_range(1:11)))
% set(gca, 'YTick', (1/num_pts:1/num_pts:1)*num_pts, 'YTickLabel', unique_names1)

load('\all_features.mat','cpc_scores','unique_names');
cpc_labels = zeros(size(labels));

for i = 1:length(labels)
    indx = ~cellfun(@isempty, strfind(unique_names,unique_names1{i}));
    indx = find(indx==1);
    indx = indx(1);
    cpc_labels(i) = cpc_scores(indx);
end
figure,
tmp = feature_all1(cpc_labels==1,1:11);
[h1,p] = boundedline(1:11, nanmean(tmp), nanstd(tmp)./sum(~isnan(tmp)),'-ro','alpha');% --b* --ro --g+ --cs
% h1 = outlinebounds(l,p);
hold on,
tmp = feature_all1(cpc_labels==2,1:11);
[h2,p] = boundedline(1:11, nanmean(tmp), nanstd(tmp)./sum(~isnan(tmp)),'-b*','alpha');% --b* --ro --g+ --cs
% h2 = outlinebounds(l,p);
hold on,
tmp = feature_all1(cpc_labels==3,1:11);
[h3,p] = boundedline(1:11, nanmean(tmp), nanstd(tmp)./sum(~isnan(tmp)),'-g+','alpha');% --b* --ro --g+ --cs
% h3 = outlinebounds(l,p);
hold on,
tmp = feature_all1(cpc_labels==4,1:11);
if ~isempty(find(sum(~isnan(tmp))==1))
    tmp(:,find(sum(~isnan(tmp))==1)) = nan;
end
[h4,p] = boundedline(1:11, nanmean(tmp), nanstd(tmp)./sum(~isnan(tmp)),'-cs','alpha');% --b* --ro --g+ --cs --md
% h4 = outlinebounds(l,p);
hold on,
tmp = feature_all1(cpc_labels==5,1:11);
[h5,p] = boundedline(1:11, nanmean(tmp), nanstd(tmp)./sum(~isnan(tmp)),'-k^','alpha');% --b* --ro --g+ --cs
% h5 = outlinebounds(l,p);
set(gca, 'XTick', 1:length(time_range(1:11)), 'XTickLabel', strsplit(num2str(time_range(1:11))));
xlabel('Hours After Cardiac Arrest');
ylabel('Mean Outcome Prediction Probability');
xlim([0 12]);
title('Mean Outcome Prediction Probability at Different Time Intervals')
legend([h5,h4,h3,h2,h1],'CPC =5','CPC = 4','CPC = 3','CPC = 2','CPC = 1');
legend boxoff
