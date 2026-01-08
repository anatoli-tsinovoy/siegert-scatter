function y = medfilt1(x, n)
% Simple 1D median filter for Octave compatibility
% x: input signal (can be matrix, filters along columns)
% n: window size
%
% This is a basic implementation to replace MATLAB's medfilt1 function

if nargin < 2
    n = 3;
end

% Handle both row and column vectors
was_row = isrow(x);
if was_row
    x = x(:);
end

[m, cols] = size(x);
y = zeros(size(x));

half_win = floor(n/2);

for col = 1:cols
    for i = 1:m
        % Determine window bounds
        idx_start = max(1, i - half_win);
        idx_end = min(m, i + half_win);
        
        % Apply median filter
        y(i, col) = median(x(idx_start:idx_end, col));
    end
end

if was_row
    y = y.';
end

end
