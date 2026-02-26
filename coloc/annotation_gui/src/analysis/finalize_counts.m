function count = finalize_counts(p,confidence)
% Custom function for filtering and finalizing Ctip2/Cux1 counts

count = zeros(size(p));

% 1) Initial assignment
for i = 1:length(confidence)
    idx = p(:,i) > confidence(i);
    count(idx,i) = 1;
    p(idx,:) = 0;
end

% 2) If one marker greater than 0.1 over the other, assign it to that
% marker
idx = p(:,2) - p(:,3) > 0.25;
count(idx,2) = 1;
p(idx,:) = 0;
idx = p(:,3) - p(:,2) > 0.25;
count(idx,3) = 1;
p(idx,:) = 0;





% 3) For the remaining cells, assign to background or both designations
% (whichever is greater)
if size(p,2) == 4
    idx = p(:,1) > p(:,4);
    count(idx,1) = 1;
    idx = p(:,1) < p(:,4);
    count(idx,4) = 1;
end

end