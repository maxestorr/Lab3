tests = zeros(4);
for i = 1:4
    tests(i,i) = nan;
    for j = 1:4 && j~=i
        tests(i,j) = ttest2(data(i,:),data(j,:))
    end
end