function threshold = calculate_threshold(values, config)
%--------------------------------------------------------------------------
% Calculate threshold from defined value or regular expression
%--------------------------------------------------------------------------

if isempty(config.expression)
    return
end

% Deconstruct regular expression
splitstring = regexp(config.expression,'\s','split');

threshold = 0;
for i = 1:length(splitstring)
    s1 = regexp(splitstring(i),'*','split');
    
    % Constant to multiply by
    if length(s1) < 2
        c = 1;
    else
        c = str2double(s1(1));
        s1(1) = [];
    end
    
    % Some stat from data
    s2 = regexp(s1,'(\w+)','match');
    if isempty(s2)
        continue
    end
    switch s2(1)
        case 'mean'
            stat = mean(values);
        case 'median'
            stat = median(values);
        case 'mode'
            stat = mode(values);
        case 'sd'
            stat = sd(values);
        case 'mad'
            stat = mad(values,1);
        case 'prctile'
            stat = prctile(values,str2double(s2(2)));
        case '+'
            continue
        otherwise
            warning("Could not recognize regular expression term %d\n",i)
    end
    % Add to threshold value
    threshold = threshold + stat*c;
end

end