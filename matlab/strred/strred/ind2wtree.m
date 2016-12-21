function  wtree = ind2wtree(pyr, ind)

%this function is called by vifvec.m
% converts the output of Eero Simoncelli's pyramid routines into subbands in a cell array
C=pyr;
S=ind;

offset=0;
numsubs=size(ind,1);
for i=1:numsubs
    wtree{numsubs-i+1}=reshape(C(offset+1:offset+prod(S(i,:))), S(i,1),S(i,2));
    offset=offset+prod(S(i,:));
end
