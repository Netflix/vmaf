function [ss ent] = est_params(y,blk,sigma)
% 'ss' and 'ent' refer to the local variance parameter and the
% entropy at different locations of the subband
% y is a subband of the decomposition, 'blk' is the block size, 'sigma' is
% the neural noise variance

sizeim=floor(size(y)./blk)*blk; % crop  to exact multiple size
y=y(1:sizeim(1),1:sizeim(2));

%Estimate the covariance matrix
temp=[];
for u=1:blk
for v=1:blk
    temp=cat(1,temp,reshape(y(v:end-(blk-v), u:end-(blk-u)),1,[]));
end
end

mcu=mean(temp,2);
cu=((temp-repmat(mcu,1,size(temp,2)))*(temp-repmat(mcu,1,size(temp,2)))')./size(temp,2);

[Q,L] = eig(cu);
L = diag(diag(L).*(diag(L)>0))*sum(diag(L))/(sum(diag(L).*(diag(L)>0))+(sum(diag(L).*(diag(L)>0))==0));
cu = Q*L*Q';

temp=[];
for u=1:blk
    for v=1:blk
        temp=cat(1,temp,reshape(y(v:blk:end, u:blk:end),1,[]));
    end
end

%Estimate local variance parameters
if max(eig(cu)) > 0
    ss=(inv(cu)*temp);
    ss=sum(ss.*temp)./(blk*blk);
    ss = reshape(ss,sizeim/blk);
else ss=zeros(sizeim(1)/blk,sizeim(2)/blk);
end

[V,d]=eig(cu); d = d(d>0); 

%Compute entropy
temp=zeros(size(ss));
for u=1:length(d)
    temp = temp+log2(ss*d(u)+sigma)+log(2*pi*exp(1));
end
ent = temp;