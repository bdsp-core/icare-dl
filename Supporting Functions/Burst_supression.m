function [ bur, sup ] = Burst_supression( x, Fs )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
supression_threshold = 10;%10;

% DETECT EMG ARTIFACTS.
% [be ae] = butter(6, [30 49]./(Fs/2)); % bandpass filter
% demg=filtfilt(be,ae,x);
% i0=1; i1=1; ct=0; dn=0;
% Nt = size(x,2);
% chunkSize = 5; % 5 second chunks
% a = zeros(1,Nt);
% while ~dn
%     %% get next data chunk
%     i0=i1;
%     if i1 == Nt
%         dn=1;
%     end
%     
%     i1=i0+round(Fs*chunkSize);
%     i1=min(i1,Nt);
%     i01=i0:i1;
%     ct=ct+1; % get next data chunk
%     
%     A(ct)=0; % set to 1 if artifact is detected
%     de=demg(:,i01);
%     
%     %% check for emg artifact
%     v=std(de);
%     if v > 5
%         A(ct)=1;
%     end
%     a(i01)=A(ct);
% end

% [c,l]=wavedec(x,9,'db2');
% A9=wrcoef('a',c,l,'db2',9);
% 
% [c,l]=wavedec(x,10,'db2');
% A10=wrcoef('a',c,l,'db2',10);
% 
% [c,l]=wavedec(x,11,'db2');
% A11=wrcoef('a',c,l,'db2',11);
% 
% x_tmp = x - A9;

% CALCULATE ENVELOPE
x = smooth(x,10);
ME = abs(hilbert(x));
% ME = smooth(e,Fs/4); % apply 1/2 second smoothing
% e = ME;
ME_temp = sort(ME);
baseline = mean(ME_temp(1:Fs*1.5));
ME = ME-baseline;
% DETECT SUPRESSIONS
% apply threshold -- 10uv
z = (ME<supression_threshold);
% remove too-short suppression segments
z = fcnRemoveShortEvents(z,Fs/4);
% remove too-short burst segments
b = fcnRemoveShortEvents(1-z,Fs/4);
z = 1-b;
z = z';

%% RUN 'BS' ALGORITHM
went_low  = find((z(1:end-1) == 0) & (z(2:end) == 1));
went_high  = find((z(1:end-1) == 1) & (z(2:end) == 0));

if ~isempty(went_low)&&~isempty(went_high)
starting = went_high(1) < went_low(1);

if(starting == 0)
    bur =  [[1, went_high(1:length(went_low)-1)]; went_low]';
    sup = [went_low(1:length(went_high)); went_high]';
end

if(starting == 1)
    sup =  [[1, went_low(1:length(went_high)-1)]; went_high]';
    bur = [went_high(1:length(went_low)); went_low]';
end

else
    bur = [];
    sup = [];

end

