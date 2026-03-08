clear all
load("Mirflickr.mat")


dataset =data;
partial_target = full(partial_target);
target = full(target);



prop_train = 0.8;  %train-test split
T = 50;           % max number of iterations
noise = 0.0;       % single noise level, change as needed
c_1=0.5;           % the constant of the treshold
rng(41);

% --- Generate partial labels ---
 partial_target = real_partial_datasets_with_noise(data, target, partial_target, noise);

% --- Split ---
[train, test] = dataset_split_part_labels(data, partial_target, target, prop_train);

% --- Preprocess ---
[train, test] = nn_preprocess_real_partial_datasets(train, test);

% --- Predict ---
pred = PL_Aknn_classifier(train, test, T, c_1);

% --- Accuracy ---
acc = mean(arrayfun(@(i) test.target(pred(i), i) > 0, 1:size(test.data, 1)));
fprintf("Noise %.1f: error = %.4f\n", noise, 1 - acc);
