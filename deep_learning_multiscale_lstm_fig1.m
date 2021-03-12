%% plot roc curves
clear
close all
load('\multi-lstm-summary-clinical_avg.mat','mean_curve_all','mean_curve_seperate_all','auc_micro','auc_micro_std','prob_all','label_all');

time_step = 6;
time_range = 12:time_step:72;
intervals= linspace(0, 1, 100);
colormaps = [{'[128,128,128]/255'};{'[0, 213, 255]/255'};{'[149, 0, 255]/255'};{'[123, 255, 0]/255'};{'[1,0,0]'};{'[0,0,1]'}];
count = 1;
for time_points = 66%12:12:72
index = find(time_range==time_points);
auc_micro(index);
mean_curve = mean_curve_all{index};
mean_curve_seperate = mean_curve_seperate_all{index};
ySEM = std(mean_curve_seperate)/sqrt(size(mean_curve_seperate,1));                              % Compute ‘Standard Error Of The Mean’ Of All Experiments At Each Value Of ‘x’
CI95 = tinv([0.025 0.975], size(mean_curve_seperate,1)-1);                    % Calculate 95% Probability Intervals Of t-Distribution
yCI95 = bsxfun(@times, ySEM, CI95(:));              % Calculate 95% Confidence Intervals Of All Experiments At Each Value Of ‘x’

% x is a vector, matrix, or any numeric array of data. NaNs are ignored.
% p is the confidence level (ie, 95 for 95% CI)
% The output is 1x2 vector showing the [lower,upper] interval values.
CIFcn = @(x,p)prctile(x,abs([0,100]-(100-p)/2));

auc_boostrap = [];
num_boostrap = 1000;
for iboost = 1:num_boostrap
    [labels, idx] = datasample(label_all{index},length(label_all{index}));
    preds = prob_all{index}(idx);
    [X_tmp,Y_tmp,T_tmp,AUC_tmp] = perfcurve(labels-1,preds,false);
    auc_boostrap = [auc_boostrap; AUC_tmp];
end
p = 95; 
CI = CIFcn(auc_boostrap,p);

% figure,
eval(['[l',num2str(count),', p] = boundedline(intervals, mean_curve, ySEM,''cmap'',' colormaps{count}, ',''alpha'')']);% -b r 'cmap', [123, 255, 0]/255 [149, 0, 255]/255 [0, 213, 255]/255 [128,128,128]/255
% h1 = outlinebounds(l,p);

hold on
%     

%     plot(intervals, mean_curve,'-r','linewidth',2);
    
    axis square
    box on
     grid on
    title('Average ROC Curves at Different Time Intervals')
    xlabel('1-Specificity')
    ylabel('Sensitivity')
% hold off
temp = [num2str(time_points),' h ',num2str(round(auc_micro(index),2)),' (',num2str(round(CI(1),2)),',' num2str(round(CI(2),2)),')'];
legend_disp{count} = temp;
% eval(['legend(l',num2str(index),',','''',legend_disp,'''',')'])
% legend boxoff 
count = count+1;
end
legend([l1 l2 l3 l4 l5 l6],legend_disp)
legend boxoff 

plot(intervals, intervals, 'k--')

%%
num_pts =[];
for i = 1:11
num_pts = [num_pts; size(label_all{i},1)];

end

figure, bar(num_pts);
set(gca,'xticklabel',num2cell(12:6:72));

xlabel('Hours After Cardiac Arrest');
ylabel('Number of Patients')

%% calibration curves
close all
clear
load('D:\Research\Cardiac_arrest_EEG\Codes\ComaPrognosticanUsingEEG-master\multiscale-lstm\multi-lstm-summary-clinical_avg.mat','mean_curve_all','mean_curve_seperate_all','auc_micro','auc_micro_std','prob_all','label_all');

time_step = 6;
time_range = 12:time_step:72;
intervals= linspace(0, 1, 100);
colormaps = [{'[128,128,128]/255'};{'[0, 213, 255]/255'};{'[149, 0, 255]/255'};{'[123, 255, 0]/255'};{'[1,0,0]'};{'[0,0,1]'}];
count = 1;
for time_points = 12:12:72
index = find(time_range==time_points);
yp = prob_all{index};
y = label_all{index};
y(y==2)=0;
% x is a vector, matrix, or any numeric array of data. NaNs are ignored.
% p is the confidence level (ie, 95 for 95% CI)
% The output is 1x2 vector showing the [lower,upper] interval values.
CIFcn = @(x,p)prctile(x,abs([0,100]-(100-p)/2));

M = 10;
xx = linspace(0, 1, M+1); 
yy = [];     
for k = 1:M
    y_k = y(yp>=xx(k) & yp<xx(k+1));
    yy(k) = sum(y_k)/length(y_k);
end
xx = linspace(0, 1, M+1);
xx = (xx(1:M)+xx(2:end))/2;
Brier_mean(index) = nansum(abs(xx-yy))/sum(~isnan(yy)); %% Brier's Score

hold on

eval(['l', num2str(count),' = plot(xx, yy,''Color'',',colormaps{count}, ',''linewidth'',2);']);
xx = linspace(0, 1, M+1); 
Brier_boostrap = [];
num_boostrap = 1000;
for iboost = 1:num_boostrap
    [labels, idx] = datasample(label_all{index},length(label_all{index}));
    yp = prob_all{index}(idx);
    yy = [];
    xx = linspace(0, 1, M+1); 
for k = 1:M
    y_k = y(yp>=xx(k) & yp<xx(k+1));
    yy(k) = sum(y_k)/length(y_k);
end
xx = linspace(0, 1, M+1);
xx = (xx(1:M)+xx(2:end))/2;
Brier_boostrap(iboost) = nansum(abs(xx-yy))/sum(~isnan(yy));

end
p = 95; 
CI = CIFcn(Brier_boostrap,p);

    
    axis square
    box on
    grid on
    title('Calibration')
    xlabel('Prediction')
    ylabel('Proportion of the positive')

temp = [num2str(time_points),' h ',num2str(round(Brier_mean(index),2)),' (',num2str(round(CI(1),2)),',' num2str(round(CI(2),2)),')'];
legend_disp{count} = temp;
% eval(['legend(l',num2str(index),',','''',legend_disp,'''',')'])
% legend boxoff 
count = count+1;
end
xx = linspace(0, 1, M+1); 
plot(xx, xx, 'k--')

legend([l1 l2 l3 l4 l5 l6],legend_disp)
legend boxoff 

