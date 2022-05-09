%% Data Processing
clear all; close all; clc;

load("contdata95.mat");

% Channels with 80% Contribution to MSE
ind = [22 24 27 30 31 41 47 53 63 65 78 84 86 89 95];

% Change to Add History
enable_history = false;

if enable_history == true

    history = 4;
    [n, n_neurons] = size(Y);
    
    Y_delay = zeros(n - history, n_neurons*(history + 1) + 1);
    
    for i = 1:size(Y_delay,1)
        Y_delay(i,:) = [1, reshape(Y(i:i+history,:),1,[])];
    end
    
    n_delay = size(Y_delay,1);
    % Splitting into Training & Test Data
    
    train_X = X(history+1:floor(n_delay/2)+history,:)';
    train_Y = Y_delay(1:floor(n_delay/2),ind)';
    test_X = X(floor(n_delay/2)+history+1:n_delay+history,:)';
    test_Y = Y_delay(floor(n_delay/2)+1:n_delay,ind)';

else
%     X = [X, ones(size(X,1),1)];
    [n, n_neurons] = size(Y);
    n_train = round(0.7*n);
    n_validation = round(0.15*n);
    train_X = X(1:n_train,:)';
    test_X = X(n_train+n_validation+1:end,:)';
    train_Y = Y(1:n_train,ind)';
    test_Y = Y(n_train+n_validation+1:end,ind)';
end
%% Calculating Regressors

% Initializing Variables
n = size(train_X,2);
n_st = size(train_X,1);

% Non-Linear Observation Model
h = @(X,B) B*[X(1,:); X(2,:); sqrt(X(1,:).^2 + X(2,:).^2); X(3,:); X(4,:); sqrt(X(3,:).^2 + X(4,:).^2)];
H = @(X,B) [B(:,1) + B(:,3)*X(1,:)/sqrt(X(1,:)^2 + X(2,:)^2), ... 
            B(:,2) + B(:,3)*X(2,:)/sqrt(X(1,:)^2 + X(2,:)^2), ...
            B(:,4) + B(:,6)*X(3,:)/sqrt(X(3,:)^2 + X(4,:)^2), ...
            B(:,5) + B(:,6)*X(4,:)/sqrt(X(3,:)^2 + X(4,:)^2)];

% Prediction Model Parameters
Xt = train_X(:,2:end);
Xt_1 = train_X(:,1:end-1);
Yt = train_Y(:,2:end);

A = Xt*Xt_1'/(Xt_1*Xt_1');
C = Yt*Xt'/(Xt*Xt');

% Observation Model Parameters
xB = [train_X(1,:); train_X(2,:); sqrt(train_X(1,:).^2 + train_X(2,:).^2); ...
      train_X(3,:); train_X(4,:); sqrt(train_X(3,:).^2 + train_X(4,:).^2)];
B = train_Y*xB'/(xB*xB');

% Noise Covariances
W = 1/(n-1)*(Xt - A*Xt_1)*(Xt - A*Xt_1)';
Q = 1/n*(Yt - C*Xt)*(Yt - C*Xt)';

%% Kalman Filter

% Initializing
xt = test_X(:,1);
Pt = W;
Yt = test_Y;

KGain_kf = zeros(4, 15, length(test_X)-1);

pred_X = zeros(size(test_X));
pred_X(:,1) = xt;

for t = 2:length(test_X)
    % Predict
    xt_hat = A*xt;
    Pt_hat = A*Pt*A' + W;
    
    % Innovate
    Kt = Pt_hat*C'/(C*Pt_hat*C' + Q);
    KGain_kf(:,:,t-1) = Kt;
    
    % Update
    xt = xt_hat + Kt*(Yt(:,t) - C*xt_hat);
    Pt = (eye(size(Kt,1)) - Kt*C)*Pt_hat;
    
    pred_X(:,t) = xt;
end

% MSE & Correlation
disp('Kalman Filter')
MSE = mean((pred_X' - test_X').^2, 'all')
sigma = diag(corr(pred_X', test_X'))'
disp(repmat('-',[1,40]));

%% EKF

% Initializing
xt = test_X(:,1);
Pt = W;
Yt = test_Y;

pred_X = zeros(size(test_X));
pred_X(:,1) = xt;

for t = 2:length(test_X)
    % Predict
    xt_hat = A*xt;
    Pt_hat = A*Pt*A' + W;

    % Innovate
    Ht = H(xt_hat,B);
    Kt = Pt_hat*Ht'/(Ht*Pt_hat*Ht' + Q);

    % Update
    xt = xt_hat + Kt*(Yt(:,t) - h(xt_hat,B));
    Pt = (eye(size(Kt,1)) - Kt*Ht)*Pt_hat;
    
    pred_X(:,t) = xt;
end

% MSE & Correlation
disp('EKF')
MSE = mean((pred_X' - test_X').^2, 'all')
sigma = diag(corr(pred_X', test_X'))'
disp(repmat('-',[1,40]));

%% UKF

% Initializing
xt = test_X(:,1);
Pt = W; 
Yt = test_Y;

pred_X = zeros(size(test_X));
pred_X(:,1) = xt;

% Tuneable Kappa
k = 0.01;

for t = 2:length(test_X)
    % Predict
    xt_hat = A*xt;
    Pt_hat = A*Pt*A' + W;

    % Generating Sigma Points
    L = sqrt(n_st + k)*chol(Pt_hat, 'lower');
    X_sigma = [xt_hat, xt_hat + L, xt_hat - L];

    % Initializing Weights
    w = [k/(n_st+k); 1/(2*(n_st+k))*ones(2*n_st,1)];

    % Weighted Mean & Covariances of Observations
    Yt_hat = h(X_sigma,B);
    yt_hat = Yt_hat*w;
    Pzz = (Yt_hat - yt_hat)*diag(w)*(Yt_hat - yt_hat)' + Q;
    Pxz = (X_sigma - xt_hat)*diag(w)*(Yt_hat - yt_hat)';

    % Filter Gain
    Kt = Pxz/Pzz;

    % Update
    xt = xt_hat + Kt*(Yt(:,t) - yt_hat);
    Pt = Pt_hat - Pxz/(Pzz')*Pxz';

    pred_X(:,t) = xt;
end

% MSE & Correlation
disp('UKF')
MSE = mean((pred_X' - test_X').^2, 'all')
sigma = diag(corr(pred_X', test_X'))'
disp(repmat('-',[1,40]));

%% Run to find Channels with Max. % Contribution

percent_contr = 80;

% Establish Baseline MSE
A = (train_Y*train_Y')\train_Y*train_X';
baseline = mean((train_X' - train_Y'*A).^2, 'all');

del_ind = [];

X = train_X';
n = size(train_Y,1);
roll_mse = [];

% Removing Channels with Low Contributions
while true
    delta_mse = zeros(n,1);
    for k = 1:n
        if ismember(k,del_ind)
            delta_mse(k) = nan;
        else
            Y = train_Y';
            Y(:,[del_ind, k]) = [];
    
            A = (Y'*Y)\Y'*X;
            delta_mse(k) = mean((X - Y*A).^2, 'all') - baseline;
        end
    end
    [min_delta, i_min] = min(delta_mse);

    if min_delta/baseline*100 > (100 - percent_contr)
        break;
    else
        del_ind = [del_ind, i_min];
        roll_mse = [roll_mse, min_delta + baseline];
    end
end

ind = 1:n;
ind = ind(~ismember(ind,del_ind));
disp('Channels with ' + string(percent_contr) + '% Contribution:');
disp(ind)