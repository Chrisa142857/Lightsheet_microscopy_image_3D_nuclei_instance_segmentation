function df_agg = aggregate_by_group(df,delimiters,variable,agg)

%%%%%%%Not done



if nargin<4
    type = 'mean';
end

assert(length(variable)==1, "Can only aggregate one variable at a time") 

% Get column names
colnames = df.Properties.VariableNames;

% Subset specific variables
idx = contains(colnames,variable);
df = df(:,idx);


a = 1;






end