clear all
addpath('../code')
addpath('../data')
load("cifar_googlenet.mat")
dataset = cifar_raw;

prop_train = 0.8;  %train-test split
T = 400;           % max number of iterations
noise = 0.2;       % single noise level, change as needed
c_1=0.5;           % the constant of the treshold
num_clusters=5;    % numbers of clusters for the candidate generation
rng(42);

% --- Generate partial labels ---
[data, target, partial_target] = candidate_generator_vision(dataset, noise, num_clusters);

% --- Split ---
[train, test] = dataset_split_part_labels(data, partial_target, target, prop_train);

% --- Preprocess ---
[train, test] = nn_preprocess_vision(train, test);

% --- Predict ---
pred = PL_Aknn_classifier(train, test, T, c_1);

% --- Accuracy ---
acc = mean(arrayfun(@(i) test.target(pred(i), i) > 0, 1:size(test.data, 1)));
fprintf("Noise %.1f: error = %.4f\n", noise, 1 - acc);
