function [ TOTAL_running_TIME,RMSE,FIT_T,FIT_M,RMSE_TIME,U,Q,H,V,W,F ] = TASTE_BPP( X,A,R,conv_tol,seed,PARFOR_FLAG,normX,normA,Size_input,Constraints,mu,lambda )
%Implementation of PARACouple2 
%   


tStart=tic;
RMSE_TIME=[];
ROOTPATH = '';

J=size(X{1}, 2); %  number of features (variables)
K = max(size(X));% number of subjects
Q=cell(K,1);

U=cell(K,1);

rng(seed); % initilizing the modes based on some seed
V = rand(J,R);
W = rand(K,R);
H = rand(R);
F=rand(size(A,2),R);
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
    %t_ten=tic;
    %update Q_k
    if (PARFOR_FLAG)
        parfor k=1:K
              
               [T1,~,T2]=svd(mu*(U{k}*H),'econ');
               Q{k}=T1*T2';
        end
    else
        for k=1:K
            [T1,~,T2]=svd(mu*(U{k}*H),'econ');
            Q{k}=T1*T2';
        end
    end
    %[FIT_T FIT_M,RMSE]=calculate_fit(X,A,U,W,V,F,normX,normA,Size_input,K,PARFOR_FLAG);

    
    Q_T_U=0;
    if (PARFOR_FLAG)
        parfor k=1:K
            Q_T_U=Q_T_U+(mu*Q{k}'*U{k});
        end     
    else
        for k=1:K
            Q_T_U=Q_T_U+(mu*Q{k}'*U{k});
        end
    end
    H=Q_T_U/(K*mu);
  

    %toc(t_ten);
    %t_ten=tic;
    %update S_k
    V_T_V=V'*V;
    F_T_F=F'*F;
    if (PARFOR_FLAG)
        parfor k=1:K
         
           Khatrio_rao=diag(U{k}'*X{k}*V);
          W(k,:)=nnlsm_blockpivot( ((U{k}'*U{k}).*(V_T_V))+(lambda*F_T_F), Khatrio_rao+(lambda*F'*A(k,:)'), 1, W(k,:)' )';
        end
    else
        for k=1:K
         
           Khatrio_rao=diag(U{k}'*X{k}*V);
          W(k,:)=nnlsm_blockpivot( ((U{k}'*U{k}).*(V_T_V))+(lambda*F_T_F), Khatrio_rao+(lambda*F'*A(k,:)'), 1, W(k,:)' )';
        end
    end
    %toc(t_ten);
    
   % t_ten=tic;
    %update F
    F=nnlsm_blockpivot( lambda*W'*W, lambda*W'*A, 1, F' )';
     %toc(t_ten);
    
    %t_ten=tic;
    U_S_T_U_S=0;
    U_S_T_X=0;
    %update V
    if (PARFOR_FLAG)
        parfor k=1:K
            %U_S=(U{k}*diag(W(k,:)));%comment out
            U_S=bsxfun(@times,U{k}, W(k,:));
            U_S_T_U_S=U_S_T_U_S+U_S'*U_S;
            U_S_T_X=U_S_T_X+U_S'*X{k};
        end
    else
        for k=1:K
            %U_S=(U{k}*diag(W(k,:)));%comment out
            U_S=bsxfun(@times,U{k}, W(k,:));
            U_S_T_U_S=U_S_T_U_S+U_S'*U_S;
            U_S_T_X=U_S_T_X+U_S'*X{k};
        end
    end
    V=nnlsm_blockpivot( U_S_T_U_S, U_S_T_X, 1, V' )';
    %[FIT_T FIT_M,RMSE]=calculate_fit(X,A,U,W,V,F,normX,normA,Size_input,K,PARFOR_FLAG);
    %toc(t_ten);
    
    %t_ten=tic;
    %a=0;
    %update U_k
    
    %V_S_T_V_S=cell(K,1);
    %U_S_T_X=cell(K,1);
    if (PARFOR_FLAG)
        parfor k=1:K
             %t_ten1=tic;
            %V_S=V*(diag(W(k,:)));%comment out
            V_S=bsxfun(@times,V, W(k,:));
            V_S_T_V_S=V_S'*V_S+mu*eye(R);
            %V_S_T_V_S=sparse(V_S_T_V_S);
            U_S_T_X=V_S'*X{k}'+(mu*H'*Q{k}');
            %U_S_T_X=sparse(U_S_T_X);
            
            U{k}=nnlsm_blockpivot( V_S_T_V_S, U_S_T_X, 1, U{k}' )';
            

           
            
        end
    else
        for k=1:K
            %t_ten1=tic;
            %V_S=V*(diag(W(k,:)));%comment out
            V_S=bsxfun(@times,V, W(k,:));
            V_S_T_V_S=V_S'*V_S+mu*eye(R);
            %V_S_T_V_S=sparse(V_S_T_V_S);
            U_S_T_X=V_S'*X{k}'+(mu*H'*Q{k}');
            %U_S_T_X=sparse(U_S_T_X);
            
            U{k}=nnlsm_blockpivot( V_S_T_V_S, U_S_T_X, 1, U{k}' )';
               
        end
    end

    
    tEnd = toc(t_tennn);
    TOTAL_running_TIME=TOTAL_running_TIME+tEnd;
    prev_RMSE=RMSE;
    [FIT_T FIT_M,RMSE]=calculate_RMSE(X,A,U,W,V,F,normX,normA,Size_input,K,PARFOR_FLAG);

    
    RMSE_TIME(itr,1)=TOTAL_running_TIME;
    RMSE_TIME(itr,2)=RMSE;
     
    
    
end




end









