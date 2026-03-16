function [train, test, train_s] = dataset_split_part_labels(data, partial_target, target, prop_train)
    train = {};
    test = {};

    % ------------------------------
    % Select randomly the training samples
    % ------------------------------
    n = size(data, 1);                 % total number of samples
    k = floor(n * prop_train);         % number of training samples
    train_s = randperm(n, k);          % select k unique random indices

    % ------------------------------
    % Create training set
    % ------------------------------
    train.data = data(train_s, :);
    train.partial_target = partial_target(:, train_s);
    train.target = target(:, train_s);

    % ------------------------------
    % Create test set
    % ------------------------------
    test_s = 1:n;
    test_s(train_s) = [];              % remove training indices

    test.data = data(test_s, :);
    test.partial_target = partial_target(:, test_s);
    test.target = target(:, test_s);
end
