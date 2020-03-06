function [ fit_tensor, fit_matrix,RMSE ] = calculate_RMSE( X,A,U,W,V,F,normX,normA,Size_input,K,PARFOR_FLAG )
%Calculate fit for parafac2 problem
    RMSE=0;
    fit_tensor=0;
    fit_matrix=0;
    if (PARFOR_FLAG)
        parfor k = 1:K
         M   = (U{k})*diag(W(k,:))*V';
         fit_tensor = fit_tensor +norm(X{k} - M,'fro')^2;
        end
        RMSE=RMSE+fit_tensor;
        fit_tensor=1-(fit_tensor/normX);
    else
        for k = 1:K
         M   = (U{k})*diag(W(k,:))*V';
         fit_tensor = fit_tensor +norm(X{k} - M,'fro')^2;
        end
        RMSE=RMSE+fit_tensor;
        fit_tensor=1-(fit_tensor/normX);

    end
    
    %fit_matrix=1-(sum(sum( (A - (W*F') ).^2))/normA);
    RMSE_mat=norm((A - (W*F') ),'fro')^2;
    RMSE=RMSE+RMSE_mat;
    %RMSE_nnz=sqrt(RMSE/num_non_z)
    RMSE=sqrt(RMSE/Size_input);
    
    fit_matrix=1-(RMSE_mat/normA);
end

