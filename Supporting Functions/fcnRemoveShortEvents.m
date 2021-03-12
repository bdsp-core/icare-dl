function z=fcnRemoveShortEvents(z,n); 

% check for too-short suppressions
ct=0; i0=1; i1=1; 
for i=2:length(z); 
    if z(i)==z(i-1); 
        ct=ct+1; i1=i; 
    else 
        if ct<n; z(i0:i1)=0; end
        ct=0; i0=i; i1=i;                
    end                
end

