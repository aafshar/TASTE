%clc;
clear all;
close all;

addpath(genpath('./TASTE_Framework'));
addpath(genpath('./nonnegfac-matlab-master')); % this package is from https://www.cc.gatech.edu/~hpark/nmfsoftware.php



%%create a synthetic data set. 
K=100;
J=40;%number of dynamic features
P=30;%number of static features
I_k=50;
R=4;  %number of factors or components
H=rand(R,R);
W=rand(K,R);
F=rand(P,R);
V=rand(J,R);
Q=cell(K,1);
U=cell(K,1);

for k=1:K
    col_Q_k=randi(R,I_k,1);
    Temp_Q=repmat(col_Q_k,1,R);
    for r=1:R
        col_Q=Temp_Q(:,r)==r;
        Q{k}(:,r)=col_Q;
    end  
    Q{k}=normc(Q{k});
%Q{k}=orth(rand(I_k,R)); 

    U{k}=Q{k}*H;
end

A=W*F';
X = cell(K,1);
for i=1: K
        X{i}=(U{i}*diag(W(i,:)))*V';
end




data_name="Synethetic_data";



lambda=1; 
mu=1;
conv_tol=1e-5; %converegance tolerance
PARFOR_FLAG=0; %parallel computing
%parpool('local',30)
[normX,normA,Size_input]=claculate_norm(X,A,K,PARFOR_FLAG); %Calculate the norm of the input X
Constraints={'nonnegative', 'nonnegative','nonnegative','nonnegative'};

itr=5;
seed=1;

[TOTAL_running_TIME,rmse,FIT_Tensor,FIT_Matrix,RMSE_TIME,U,Q,H,V,W,F]=TASTE_BPP(X,A,R,conv_tol,seed,PARFOR_FLAG,normX,normA,Size_input,Constraints,mu,lambda);
plot(RMSE_TIME(:,1),RMSE_TIME(:,2));
xlabel("Time");
ylabel("RMSE");

