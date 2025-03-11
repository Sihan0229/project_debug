function Z=flatten(X,ind)

sz=size(X);

if isnumeric(ind)
  nd=max(ndims(X),ind);
  ind = {ind, [ind+1:nd, 1:ind-1]};
else
  nd=max(cellfun(@max,ind));
end


if length(ind{1})~=1 || ind{1}~=1
  X=permute(X,cell2mat(ind));
end

if length(ind{1})==1
  Z=X(:,:);
else
  Z=reshape(X,[prod(sz(ind{1})),prod(sz(ind{2}))]);
end