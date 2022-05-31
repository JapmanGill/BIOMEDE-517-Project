clear;
close all;

%%
runFile = 'G:\My Drive\UMich\Data\KalmanNet\Z_Joker_2021-04-12_Run-003.mat';
binSize = 32; % Number of ms to bin data into
featList = {'FingerAnglesTIMRL', 'NeuralFeature'}; % For use in getZFeats.m - these are the features we want from the z struct

z = load(runFile);
z = z.z; % The load function puts all the variables in a struct by default. Pretty sure you can call load(runFile,'z') instead.

disp("Getting binned features")
feat = getZFeats(z, binSize, 'featList', featList); % Get features in binSize ms bins

%% Transform to X and Y matrices
% X contains pos and speed of finger flexions
X = [feat{1}(:, 2) feat{1}(:, 4) feat{1}(:, 7) feat{1}(:, 9)];
% Y contains all neurons
Y = feat{2};

%% Extract train, validation and test
bins_train = round(size(Y, 1) * 0.7);
bins_validation = bins_train + round(size(Y, 1) * 0.15);
Y_train = Y(1:bins_train, :);
X_train = X(1:bins_train, :);
%% Find N best channels
% Baseline MSE
B = (Y_train' * Y_train) \ Y_train' * X_train;
baseline = mean((X_train - Y_train * B).^2, 'all')
num_to_keep = 20;
total_channels = width(Y);
channels_to_keep = 1:total_channels;

% For each iteration, remove the channel that has the lowest impact on MSE
for i = 1:(total_channels - num_to_keep)
    mse = zeros(length(channels_to_keep), 1);

    for j = 1:length(channels_to_keep)
        c = channels_to_keep(j);
        Y_aux = Y_train(:, channels_to_keep);
        Y_aux(:, j) = [];
        B = (Y_aux' * Y_aux) \ Y_aux' * X_train;
        mse(j) = mean((X_train - Y_aux * B).^2, 'all');
    end

    [val, min_ind] = min(mse);
    fprintf("Max MSE: %f\n", val);
    channels_to_keep(min_ind) = [];
end

%%
Y_train = Y(1:bins_train, channels_to_keep);
Y_val = Y(bins_train + 1:bins_validation, channels_to_keep);
Y_test = Y(bins_validation + 1:end, channels_to_keep);
X_train = X(1:bins_train, :);
X_val = X(bins_train + 1:bins_validation, :);
X_test = X(bins_validation + 1:end, :);
x_0 = X_train(end, :)

B = (Y_train' * Y_train) \ Y_train' * X_train;
baseline_2 = mean((X_train - Y_train * B).^2, 'all');
fprintf("With 20 channels: %.2f%% greater MSE than baseline\n", 100 * (baseline_2 / baseline - 1));

%% Calculate linear model matrices
X_train_t = X_train(2:end, :)';
X_train_t_1 = X_train(1:end - 1, :)';
Y_train_t = Y_train(2:end, :)';
A = X_train_t * X_train_t_1' / (X_train_t_1 * X_train_t_1');
C = Y_train' * X_train / (X_train' * X_train);
W = (X_train_t - A * X_train_t_1) * (X_train_t - A * X_train_t_1)' / (size(X_train, 1) - 1);
Q = (Y_train_t - C * X_train_t) * (Y_train_t - C * X_train_t)' / (size(X_train, 1) - 1);
% Check fits
x_hat = A * X_train_t_1;
y_hat = (C * X_train')';
corr_x = diag(corr(X_train_t', x_hat'))
corr_y = diag(corr(Y_train, y_hat))
mse_x = mean((X_train_t - x_hat).^2, 2);
mse_y = mean((Y_train - y_hat).^2, 2);

%% Calculate non-linear transformation
h = @(X, B) [X(:, 1) X(:, 2) sqrt(X(:, 1).^2 + X(:, 2).^2) X(:, 3) X(:, 4) sqrt(X(:, 3).^2 + X(:, 4).^2)] * B;
% Observation Model Parameters
xB = [X_train(:, 1) X_train(:, 2) sqrt(X_train(:, 1).^2 + X_train(:, 2).^2) ...
    X_train(:, 3) X_train(:, 4) sqrt(X_train(:, 3).^2 + X_train(:, 4).^2)];
B = (xB' * xB) \ xB' * Y_train;

%% Evaluate non-linear transformation
y_train_hat = h(X_train, B);
corr_y = diag(corr(Y_train, y_train_hat))
mse_y = mean((Y_train - y_hat).^2, 1)

%% Separate into batches
% 20 time bins per sequence
bins_per_batch = 20;
length_y = length(channels_to_keep);
length_x = width(X);
% Drop the last few bins if they are not divisible by the number of bins
% per batch
last_row_train = length(X_train) - mod(length(X_train), bins_per_batch);
last_row_val = length(X_val) - mod(length(X_val), bins_per_batch);
last_row_test = length(X_test) - mod(length(X_test), bins_per_batch);
bY_train = permute(reshape(Y_train(1:last_row_train, :)', length_y, bins_per_batch, []), [3, 1, 2]);
bY_val = permute(reshape(Y_val(1:last_row_val, :)', length_y, bins_per_batch, []), [3, 1, 2]);
bY_test = permute(reshape(Y_test(1:last_row_test, :)', length_y, bins_per_batch, []), [3, 1, 2]);
bX_train = permute(reshape(X_train(1:last_row_train, :)', length_x, bins_per_batch, []), [3, 1, 2]);
bX_val = permute(reshape(X_val(1:last_row_val, :)', length_x, bins_per_batch, []), [3, 1, 2]);
bX_test = permute(reshape(X_test(1:last_row_test, :)', length_x, bins_per_batch, []), [3, 1, 2]);
bx_0 = [0.5, 0.5, 0, 0];

%% Run Kalman filter
P_t_t = W;
x_t_t = X_test(1, :)';
x_values = [x_t_t];

for i = 2:size(X_test, 1)
    x_t_t1 = A * x_t_t;
    P_t_t1 = A * P_t_t * A' + W;
    yt_tilde = Y_test(i, :)' - C * x_t_t1;
    St = C * P_t_t1 * C' + Q;
    Kt = P_t_t1 * C' / St;

    x_t_t = x_t_t1 + Kt * yt_tilde;
    x_values = [x_values x_t_t];
    P_t_t = (eye(4) - Kt * C) * P_t_t1;
end

corr_kalman = diag(corr(double(X_test), x_values'))
mse_kalman = mean((X_test - x_values').^2)
