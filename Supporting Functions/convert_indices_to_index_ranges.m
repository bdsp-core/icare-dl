function [index_ranges] = convert_indices_to_index_ranges(indices, smoothing_amt)
% Converts a vector of indices into a matrix of 
% [start_index end_index] ranges, with optional smoothing

% For example: [5 6 7 20 21 22 23 60 61] -> [5 7; 20 23; 60 61]
%   indices - vector of indices
%   smoothing_amt - maximum difference between "consecutive" indexes in a
%                   range. Default 1. For example, [5 6 8 10] with
%                   smoothing_amt = 2 gives output [5 10] range.
    if nargin < 2
        smoothing_amt = 1;
    end
    if isempty(indices)
        index_ranges = zeros(0,2);
        return;
    end
    index_ranges = [indices(1) indices(1)];
    for i=2:length(indices)
        last_index = index_ranges(end);
        current_index = indices(i);
        % If current index is within smoothing_amt samples of previous index,
        % we consider it to be in the same range. This smoothes the
        % indices a bit.
        if current_index - last_index > smoothing_amt
            % start new range
            index_ranges = [index_ranges; current_index current_index];
        else 
            % in same range
            index_ranges(end) = current_index;
        end
    end
end

