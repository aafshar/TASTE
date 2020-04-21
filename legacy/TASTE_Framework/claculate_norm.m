function [ normX,normA,Size_input ] = claculate_norm(X,A,K,PARFOR_FLAG)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    normX=0;
    Size_input=(size(A,1)*size(A,2));
    num_non_z=nnz(A);
    normA=sum(sum( (A).^2));
    if (PARFOR_FLAG)
        parfor k = 1:K
         normX = normX + sum(sum( (X{k}).^2));
         Size_input=Size_input+(size(X{k},1)*size(X{k},2));
         num_non_z=num_non_z+nnz(X{k});
        end
    else
        for k = 1:K
         normX = normX + sum(sum( (X{k}).^2));
         Size_input=Size_input+(size(X{k},1)*size(X{k},2));
         num_non_z=num_non_z+nnz(X{k});
        end
    end


end

