%GET ARMA PARAMS.
function [arma_mod] = ARMA_BIC(x,op,oq)

    %Compute the autocorrelation coefficients.
    %[d1,p1] = aryule(x(:,1),3);
    
    %Brockwell and Davis recommend using AICc for finding p-AR and q-MA

    LOGL = zeros(op,oq);
    PQ = zeros(op,oq);
    for p = 1:op
        parfor q = 1:oq
            mod = arima(p,0,q);
            [fit,~,logL] = estimate(mod,x,'print',false);
            LOGL(p,q) = logL;
            PQ(p,q) = p+q;
         end
    end
    
    %Find the values that minimize the BIC
    LOGL = reshape(LOGL,size(LOGL,1)*size(LOGL,2),1);
    PQ = reshape(PQ,size(PQ,1)*size(PQ,2),1);
    [~,bic] = aicbic(LOGL,PQ+1,100);
    bicmat = reshape(bic,op,oq);
    [p_val,q_val] =find(bicmat ==  min(min(bicmat)));
   
    %Extract the parameters for the AR_model
    mod = arima(p_val,0,q_val)
    arma_mod = estimate(mod,x)
    %armax(x,[p_val,qval])

end