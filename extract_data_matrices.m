clear;
close all;
load contdata95.mat

channels_to_keep = [22 24 27 30 31 41 47 53 63 65 78 84 86 89 95];

Y = Y(:,channels_to_keep);
% Define train and test sets
bins_train = round(size(Y,1) * 0.7);
bins_validation = bins_train + round(size(Y,1) * 0.15);
Y_train = Y(1:bins_train, :);
Y_val = Y(bins_train+1:bins_validation, :);
Y_test = Y(bins_validation+1:end,:);
X_train = X(1:bins_train,:);
X_val = X(bins_train+1:bins_validation, :);
X_test = X(bins_validation+1:end,:);

% Find matrices
X_train_t = X_train(2:end,:)';
X_train_t_1 = X_train(1:end-1,:)';
Y_train_t = Y_train(2:end,:)';
A = X_train_t*X_train_t_1'/(X_train_t_1*X_train_t_1');
C = Y_train'*X_train/(X_train'*X_train);
W = (X_train_t - A*X_train_t_1)*(X_train_t-A*X_train_t_1)' / (size(X_train,1)-1);
Q = (Y_train_t-C*X_train_t)*(Y_train_t-C*X_train_t)' / (size(X_train,1)-1);

% Check fits
x_hat = A*X_train_t_1;
y_hat = (C*X_train')';
corr_x = diag(corr(X_train_t', x_hat'))
corr_y = diag(corr(Y_train, y_hat))
mse_x = mean((X_train_t - x_hat).^2,2);
mse_y = mean((Y_train - y_hat).^2,2);

%% Run Kalman filter
P_t_t = W;
x_t_t = X_test(1,:)';
x_values = [x_t_t];
for i=2:size(X_test,1)
   x_t_t1 = A*x_t_t;
   P_t_t1 = A*P_t_t*A'+W;
   yt_tilde = Y_test(i,:)' - C*x_t_t1;
   St = C*P_t_t1*C' + Q;
   Kt = P_t_t1*C'/St;
   
   x_t_t = x_t_t1 + Kt*yt_tilde;
   x_values = [x_values x_t_t];
   P_t_t = (eye(4) - Kt*C)*P_t_t1;
end

corr_kalman = diag(corr(X_test, x_values'))
mse_kalman = mean((X_test- x_values').^2)
