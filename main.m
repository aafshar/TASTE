%clc;
clear all;
close all;

addpath(genpath('./TASTE_Framework'));
addpath(genpath('./nonnegfac-matlab-master')); % this package is from https://www.cc.gatech.edu/~hpark/nmfsoftware.php

R = 20;

% Case need to be fit before projection!!

if isfile("case.mat")
    load("case.mat", 'A', 'X', 'K', 'P', 'X_height', 'J');
else
    A = readtable('static.csv', 'HeaderLines', 1);
    sele = A{:, end};
    sele = sele(:, :);
    A = A(sele==1, :);
    XK = readtable('dynamic.csv', 'HeaderLines', 1);
    [K, P] = size(A);
    X_height = height(XK);
    J = max(XK{:, 3});
    X = cell(K, 1);
    j = 1;
    for k = 1:K
        while (j <= X_height) && ~strcmp(A{k, 1}{1}, XK{j, 1}{1})
            j = j + 1;
        end
        start = j;
        while (j <= X_height) && strcmp(A{k, 1}{1}, XK{j, 1}{1})
            j = j + 1;
        end
        X{k} = XK{start : (j-1), 2:3};
    end
    A = A(:, 2:end);
    for k = 1:K
        hei = size(X{k});
        X{k} = sparse(X{k}(:, 1), X{k}(:, 2), ones(hei(1), 1), X{k}(end, 1), J);
    end
    A = A{:, :};
    save("case.mat")
end

%A = ones(12494, 1);

if isfile(strcat(num2str(R), "_case.mat"))
    load(strcat(num2str(R), "_case.mat"));
else
    data_name="CMS";
    lambda=1;
    mu=1;
    conv_tol=1e-5; %converegance tolerance
    PARFOR_FLAG=0; %parallel computing
    [normX,normA,Size_input]=claculate_norm(X,A,K,PARFOR_FLAG); %Calculate the norm of the input X
    Constraints={'nonnegative', 'nonnegative','nonnegative','nonnegative'};

    itr=5;
    seed=1;

    [TOTAL_running_TIME,rmse,FIT_Tensor,FIT_Matrix,RMSE_TIME,U,Q,H,V,W,F]=TASTE_BPP(X,A,R,conv_tol,seed,PARFOR_FLAG,normX,normA,Size_input,Constraints,mu,lambda);
    figure();
    plot(RMSE_TIME(:,1),RMSE_TIME(:,2));
    xlabel("Time");
    ylabel("RMSE");
    saveas(gcf,num2str(R),'epsc');

    save(strcat(num2str(R), "_case.mat"));
end

if isfile("ctrl.mat")
    load("ctrl.mat", 'A', 'X', 'K', 'P', 'X_height', 'J');
else
    A = readtable('static.csv', 'HeaderLines', 1);
    sele = A{:, end};
    sele = sele(:, :);
    A = A(sele==0, :);
    XK = readtable('dynamic.csv', 'HeaderLines', 1);
    [K, P] = size(A);
    X_height = height(XK);
    J = max(XK{:, 3});
    X = cell(K, 1);
    j = 1;
    for k = 1:K
        while (j <= X_height) && ~strcmp(A{k, 1}{1}, XK{j, 1}{1})
            j = j + 1;
        end
        start = j;
        while (j <= X_height) && strcmp(A{k, 1}{1}, XK{j, 1}{1})
            j = j + 1;
        end
        X{k} = XK{start : (j-1), 2:3};
    end
    A = A(:, 2:end);
    for k = 1:K
        hei = size(X{k});
        X{k} = sparse(X{k}(:, 1), X{k}(:, 2), ones(hei(1), 1), X{k}(end, 1), J);
    end
    A = A{:, :};
    save("ctrl.mat")
end

%A = ones(12494, 1);

if isfile(strcat(num2str(R), "_ctrl.mat"))
    load(strcat(num2str(R), "_ctrl.mat"));
else
    data_name="CMS";
    lambda=1;
    mu=1;
    conv_tol=1e-5; %converegance tolerance
    PARFOR_FLAG=0; %parallel computing
    [normX,normA,Size_input]=claculate_norm(X,A,K,PARFOR_FLAG); %Calculate the norm of the input X
    Constraints={'nonnegative', 'nonnegative','nonnegative','nonnegative'};

    itr=5;
    seed=1;

    [ TOTAL_running_TIME,RMSE,FIT_T,FIT_M,RMSE_TIME,U,Q,H,V,W,F ] = PARACoupl2_BPP( X,A,V,F,H,R,conv_tol,seed,PARFOR_FLAG,normX,normA,Size_input,Constraints,mu,lambda );
    figure()
    plot(RMSE_TIME(:,1),RMSE_TIME(:,2));
    xlabel("Time");
    ylabel("RMSE");
    saveas(gcf,strcat(num2str(R), "_projection"),'epsc');

    save(strcat(num2str(R), "_ctrl.mat"));
end
