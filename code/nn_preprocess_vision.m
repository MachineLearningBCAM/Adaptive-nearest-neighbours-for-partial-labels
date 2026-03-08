function [train, test] = nn_preprocess_vision(train, test)
% ============================================================
% NN Preprocessing for Vision Datasets
% CENTER + PCA(50) + L2 + SMOOTH (local bw) + LOCAL DENSITY POINT TRANSFORM
% All statistics computed on training set only (no test leakage).
% ============================================================
if nargin < 3, opts = struct(); end
opts = set_default(opts, 'pca_k',       50);
opts = set_default(opts, 'mix_k',       10);
opts = set_default(opts, 'mix_alpha',   0.25);
opts = set_default(opts, 'smooth_temp', 1.0);
opts = set_default(opts, 'ls_k',        50);

model = struct();
Xtr = double(train.data);
Xte = double(test.data);
nTe = size(Xte, 1);
nTr = size(Xtr, 1);

%% 1. CENTERING
model.mean = mean(Xtr, 1);
Xtr = Xtr - model.mean;
Xte = Xte - model.mean;

%% 2. PCA
pca_k = min(opts.pca_k, size(Xtr, 2));
[model.pca_V, ~] = eigs(cov(Xtr), pca_k);
Xtr = Xtr * model.pca_V;
Xte = Xte * model.pca_V;

%% 3. L2 NORMALISATION
Xtr = l2norm(Xtr);
Xte = l2norm(Xte);

%% 4. GAUSSIAN-WEIGHTED KNN SMOOTHING
[idx_tr, dist_tr] = knnsearch(Xtr, Xtr, 'K', opts.mix_k + 1);
idx_tr  = idx_tr(:,  2:end);
dist_tr = dist_tr(:, 2:end);
sigma_tr = median(dist_tr, 2) * opts.smooth_temp + 1e-12;
W        = exp(-dist_tr.^2 ./ (2 * sigma_tr.^2));
W        = W ./ (sum(W, 2) + 1e-12);
Xtr_s = (1 - opts.mix_alpha) * Xtr;
for i = 1:nTr
    Xtr_s(i,:) = Xtr_s(i,:) + opts.mix_alpha * (W(i,:) * Xtr(idx_tr(i,:),:));
end
Xtr_s = l2norm(Xtr_s);

k_s = min(opts.mix_k, nTr);
[idx_te, dist_te] = knnsearch(Xtr_s, Xte, 'K', k_s);
sigma_te = median(dist_te, 2) * opts.smooth_temp + 1e-12;
W_te     = exp(-dist_te.^2 ./ (2 * sigma_te.^2));
W_te     = W_te ./ (sum(W_te, 2) + 1e-12);
Xte_s = (1 - opts.mix_alpha) * Xte;
for i = 1:nTe
    Xte_s(i,:) = Xte_s(i,:) + opts.mix_alpha * (W_te(i,:) * Xtr_s(idx_te(i,:),:));
end
Xte_s = l2norm(Xte_s);

%% 5. LOCAL DENSITY POINT TRANSFORM
ls_k = min(opts.ls_k, nTr - 1);

[~, D_tr_ls] = knnsearch(Xtr_s, Xtr_s, 'K', ls_k + 1);
model.rk_tr  = mean(D_tr_ls(:, 2:end), 2) + 1e-12;

[~, D_te_ls] = knnsearch(Xtr_s, Xte_s, 'K', ls_k);
rk_te        = mean(D_te_ls, 2) + 1e-12;

% divide by local density — no L2 norm after, it would undo the scaling
Xtr_w = Xtr_s ./ model.rk_tr;
Xte_w = Xte_s ./ rk_te;

%% OUTPUT
train.data = Xtr_w;
test.data  = Xte_w;
model.opts = opts;
end

% ================================================================
function Xn = l2norm(X)
    Xn = X ./ (sqrt(sum(X.^2, 2)) + 1e-12);
end

function opts = set_default(opts, field, value)
    if ~isfield(opts, field)
        opts.(field) = value;
    end
end