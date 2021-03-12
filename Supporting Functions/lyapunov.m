function lam = lyapunov(x,Fs)
% calculate lyapunov coefficient of time series

%x = sin(1:dt:10)'; % this is 68.83
%x = (1:dt:10)';    % this is 0
%x = randn(1001,1); % this is ~300
%x = squeeze(EEG.data(1,1:1001,1))'; %this is 241.
dt = 1/Fs;
[ndata nvars]=size(x);
steps_forward = 5;
N2 = floor(ndata/2);
N4 = floor(ndata/4);
TOL = 1.0e-6;

exponent = zeros(N4+1,1);
tic
for i=N4:N2  % second quartile of data should be sufficiently evolved
   
   %get all points but this one.
   js = setdiff(1:ndata-steps_forward,i);
   %find the index of the nearest neighbor
   [idx] = knnsearch(x(js,:),x(i,:));
   indx = js(idx);
   
%   tic
%    dist = norm(x(i+1,:)-x(i,:));
%    indx = i+1;
%    for j=i:ndata-5
%        if (i ~= j) && norm(x(i,:)-x(j,:))<dist
%            dist = norm(x(i,:)-x(j,:));
%            indx = j; % closest point!
%        end
%    end
%    toc
   expn = 0.0; % estimate local rate of expansion (i.e. largest eigenvalue)
   for k=1:steps_forward
       if norm(x(i+k,:)-x(indx+k,:))>TOL && norm(x(i,:)-x(indx,:))>TOL
           expn = expn + (log(norm(x(i+k,:)-x(indx+k,:)))-log(norm(x(i,:)-x(indx,:))))/k;
       end
   end
   exponent(i-N4+1)=expn/steps_forward;   
end
toc

sum=0;  % now, calculate the overal average over N4 data points ...
for i=1:N4+1
    sum = sum+exponent(i);
end

lam=sum/((N4+1)*dt);  
% return the average value
% if lam > 0, then system is chaotic
