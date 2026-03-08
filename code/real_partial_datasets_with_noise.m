function partial_target_new = real_partial_datasets_with_noise(data, target, partial_target, noise)
%  randomly removes the true label from the candidate set
%
% Inputs:
%   data           - [n x d] features (not used, kept for interface)
%   target         - [num_classes x n] one-hot true labels
%   partial_target - [num_classes x n] candidate label matrix
%   noise          - probability to remove the true label
%
% Output:
%   partial_target_new - updated candidate label matrix

[num_classes, num_samples] = size(partial_target);
partial_target_new = partial_target;

for i = 1:num_samples
    if rand() < noise
        % Find the index of the true label
        true_label_idx = find(target(:, i));
        
        % Remove the true label
        partial_target_new(true_label_idx, i) = 0;
    end
end

end
