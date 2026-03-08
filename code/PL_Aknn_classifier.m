function pred = PL_Aknn_classifier(train, test, maxK, A_0)
    %%% Preprocessing %%%%%
    delta = 0.1;
    num_labels = size(test.target, 1);
    num_test = size(test.data, 1);
    n = size(train.data, 1);
    k_a = log(n) + log(num_labels / delta);
    A = A_0 * sqrt(k_a);
    max_it = maxK;
    pred = zeros(1, num_test);

    %% NN IDENTIFICATION
    idx = knnsearch(train.data, test.data, 'K', maxK);

    for i = 1:num_test
        % Initialize candidate label set (all labels are candidates)
        threshold = ones(num_labels, 1);
        % Initialize tau (cumulative label frequencies)
        tau = zeros(num_labels, 1);
        % Initialize M matrix: M(k, y) = sqrt(k) * (Delta - (tau_y - m2) / k)
        M = inf(max_it, num_labels);
        k = 0;

        while sum(threshold) > 1 && k < max_it
            k = k + 1;
            % Compute threshold Delta
            Delta = A / sqrt(k);
            % Update tau: cumulative counts from k nearest neighbors
            tau = tau + train.partial_target(:, idx(i, k));
            % Get surviving labels
            s_hat = find(threshold);
            % Get two largest tau values among surviving labels
            tau_surviving = tau(s_hat);
            sorted_tau = sort(tau_surviving, 'descend');
            m1 = sorted_tau(1);
            if length(sorted_tau) >= 2
                m2 = sorted_tau(2);
            else
                m2 = sorted_tau(1);
            end
            % Update M matrix and eliminate labels
            for y = s_hat'
                M(k, y) = sqrt(k) * (Delta - (tau(y) - m2) / k);
                % Elimination condition
                if (m1 - tau(y)) / k >= Delta
                    threshold(y) = 0;
                end
            end
        end

        % Final label selection
        surviving_labels = find(threshold);
        if length(surviving_labels) == 1
            pred(i) = surviving_labels(1);
        else
            % Multiple labels remain: select using the heuristic criterion
            min_val = inf;
            best_label = surviving_labels(1);
            for y = surviving_labels'
                for k_iter = 1:k
                    if M(k_iter, y) < min_val
                        min_val = M(k_iter, y);
                        best_label = y;
                    end
                end
            end
            pred(i) = best_label;
        end
    end
end
