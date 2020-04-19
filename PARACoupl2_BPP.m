function [ TOTAL_running_TIME,RMSE,FIT_T,FIT_M,RMSE_TIME,U,Q,H_TASTE,V_TASTE,W,F_TASTE ] = PARACoupl2_BPP( X,A,V_TASTE,F_TASTE,H_TASTE,R,conv_tol,seed,PARFOR_FLAG,normX,normA,Size_input,Constraints,mu,lambda )
%Implementation of PARACouple2
%


tStart=tic;
RMSE_TIME=[];
ROOTPATH = '';

J=size(X{1}, 2); %  number of features (variables)
K = max(size(X));% number of subjects
Q=cell(K,1);
Q_bar=cell(K,1);
U=cell(K,1);

rng(seed); % initilizing the modes based on some seed
W = rand(K,R);
for k=1:K
    U{k}=rand(size(X{k},1),R);
end
prev_RMSE=0; RMSE=1;
itr=0;
TOTAL_running_TIME=0;

beta=1;
alpha=1;
while(abs(RMSE-prev_RMSE)>conv_tol)
    itr=itr+1;
    t_tennn=tic;
    %update Q_k
    if (PARFOR_FLAG)
        parfor k=1:K
               %I_k= size(X{k},1);
               %Q{k}=[eye(I_k);sqrt(beta)*eye(I_k);sqrt(alpha)*Q_bar{k}']\[mu*U{k}*H;sqrt(beta)*Q_bar{k};sqrt(alpha)*eye(R)];
               %Q_bar{k}=[sqrt(alpha)*Q{k}';sqrt(beta)*eye(I_k)]\[eye(R);Q{k}];

               [T1,~,T2]=svd(mu*(U{k}*H_TASTE),'econ');
               Q{k}=T1*T2';
        end
    else
        for k=1:K
            [T1,~,T2]=svd(mu*(U{k}*H_TASTE),'econ');
            Q{k}=T1*T2';
        end
    end



    %update S_k
    V_T_V=V_TASTE'*V_TASTE;
    F_T_F=F_TASTE'*F_TASTE;
    if (PARFOR_FLAG)
        parfor k=1:K
          %Khatrio_rao=efficient_khari_rao( V,U{k},X{k} ,PARFOR_FLAG);
           Khatrio_rao=diag(U{k}'*X{k}*V_TASTE);
          W(k,:)=nnlsm_blockpivot( ((U{k}'*U{k}).*(V_T_V))+(lambda*F_T_F), Khatrio_rao+(lambda*F_TASTE'*A(k,:)'), 1, W(k,:)' )';
        end
    else
        for k=1:K
          %Khatrio_rao= efficient_khari_rao( V,U{k},X{k} ,PARFOR_FLAG);
           Khatrio_rao=diag(U{k}'*X{k}*V_TASTE);
          W(k,:)=nnlsm_blockpivot( ((U{k}'*U{k}).*(V_T_V))+(lambda*F_T_F), Khatrio_rao+(lambda*F_TASTE'*A(k,:)'), 1, W(k,:)' )';
        end
    end
    %[FIT_T FIT_M,RMSE]=calculate_fit(X,A,U,W,V,F,normX,normA,Size_input,K,PARFOR_FLAG);



    %update U_k

    if (PARFOR_FLAG)
        parfor k=1:K
             %t_ten1=tic;
            %V_S=V*(diag(W(k,:)));%comment out
            V_S=bsxfun(@times,V_TASTE, W(k,:));
            V_S_T_V_S=V_S'*V_S+mu*eye(R);
            %V_S_T_V_S=sparse(V_S_T_V_S);
            U_S_T_X=V_S'*X{k}'+(mu*H_TASTE'*Q{k}');
            %U_S_T_X=sparse(U_S_T_X);

            U{k}=nnlsm_blockpivot( V_S_T_V_S, U_S_T_X, 1, U{k}' )';




        end
    else
        for k=1:K
            %t_ten1=tic;
            %V_S=V*(diag(W(k,:)));%comment out
            V_S=bsxfun(@times,V_TASTE, W(k,:));
            V_S_T_V_S=V_S'*V_S+mu*eye(R);
            V_S_T_V_S=sparse(V_S_T_V_S);
            U_S_T_X=V_S'*X{k}'+(mu*H_TASTE'*Q{k}');
            U_S_T_X=sparse(U_S_T_X);

            U{k}=nnlsm_blockpivot( V_S_T_V_S, U_S_T_X, 1, U{k}' )';


        end
    end

    tEnd = toc(t_tennn);
    TOTAL_running_TIME=TOTAL_running_TIME+tEnd;
    prev_RMSE=RMSE;
    [ FIT_T, FIT_M,RMSE ] = calculate_RMSE( X,A,U,W,V_TASTE,F_TASTE,normX,normA,Size_input,K,PARFOR_FLAG );
    % [FIT_T FIT_M,RMSE,RMSE_nnz]=calculate_fit(X,A,U,W,V_TASTE,F_TASTE,normX,normA,Size_input,num_non_z,K,PARFOR_FLAG);


    RMSE_TIME(itr,1)=TOTAL_running_TIME;
    RMSE_TIME(itr,2)=RMSE;
    % RMSE


end




end







