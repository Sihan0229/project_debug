function   X = reconstruct(u, W, eta, ModCount, sX)
    
    X1 = zeros(size(u{jjj}));
    for jjj =1:ModCount        %%%zhan kai K ci          ------2
        
         %for jj=1:K
        X1 = X1 - flatten_adj(eta*W{jjj},sX,jjj);
    end
        X=X1/(eta*ModCount);
end
