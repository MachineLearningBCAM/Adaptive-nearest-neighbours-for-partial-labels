function [data, target, partial_target] = ...
    candidate_generator_vision(dataset, noise, n_clusters)
% Candidate generator:
%  - Cluster- and label-dependent alpha
%  - With probability 'noise', true label is replaced by a random label
%  - Bag generated using alpha of the anchor label

%% ------------------------------
% Extract data and labels
%% ------------------------------
data = dataset(:,1:end-1);
labels_raw = dataset(:,end);
labels_unique = unique(labels_raw);
C = numel(labels_unique);
label_map = zeros(max(labels_raw)+1,1);
for i = 1:C
    label_map(labels_unique(i)+1) = i;
end
labels = label_map(labels_raw+1);
n = numel(labels);

%% ------------------------------
% One-hot true labels
%% ------------------------------
target = zeros(C,n);
for i = 1:n
    target(labels(i),i) = 1;
end

%% ------------------------------
% Clustering
%% ------------------------------
cluster_idx = kmeans(data, n_clusters, 'Replicates', 10);

% Alpha per (cluster, label)
alpha_cluster_label = rand(n_clusters, C)*0.8;

%% ------------------------------
% Generate partial labels
%% ------------------------------
partial_target = false(C,n);
for i = 1:n
    y = labels(i);
    c = cluster_idx(i);

    % -------- Determine anchor label --------
    if rand < noise
        anchor = randi(C);
        while anchor == y
            anchor = randi(C);  % ensure it's a wrong label
        end
    else
        anchor = y;
    end

    % -------- Use alpha of anchor label --------
    alpha_anchor = alpha_cluster_label(c, anchor);

    % Sample all labels independently with anchor alpha
    bag = rand(1,C) < alpha_anchor;

    % Ensure anchor label is always included
    bag(anchor) = true;

    partial_target(:,i) = bag';
end
end
