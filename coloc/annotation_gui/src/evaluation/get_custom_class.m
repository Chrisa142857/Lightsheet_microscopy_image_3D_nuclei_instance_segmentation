function [custom, counts] = get_custom_class(counts,single_class,custom_class)
% Combine counts from multiple classes based on specified function handle
% and append to counts matrix

custom = zeros(size(counts,1),length(custom_class));
for n = 1:length(custom_class)
    % For each channel, calculate stats
    custom_func = '@(';
    a = 1; c_idx = false(1,length(single_class));
    for i = 1:length(single_class)
        if ~contains(custom_class,strcat('CT',string(i)))
            continue
        else
            if a == 1
                custom_func = strcat(custom_func,string(strcat('CT',string(i))));
            else
                custom_func = strcat(custom_func,string(strcat(',CT',string(i))));
            end
            a = 2; c_idx(i)=1;
        end
    end
    
    % Create and apply function
    fh = str2func(strcat(custom_func,')',custom_class(n)));
    c = num2cell(counts,1);
    c = c(c_idx);
    custom(:,n) = fh(c{:});
end

% Append to counts matrix
if nargout == 2
    counts = cat(2,counts,custom);
end

end
