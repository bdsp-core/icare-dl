function [ H, prob ] = CRI_ShannonEntropy(x, bin_min, bin_max, binWidth)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[counts,binCenters] = hist(x,[bin_min:binWidth:bin_max]);
binWidth = diff(binCenters);
binWidth = [binWidth(end),binWidth]; % Replicate last bin width for first, which is indeterminate.
nz = counts>0; % Index to non-zero bins
prob = counts(nz)/sum(counts(nz));
H = -sum(prob.*log2(prob./binWidth(nz)));

end

