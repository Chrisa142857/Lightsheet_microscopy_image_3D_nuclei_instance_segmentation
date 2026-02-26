function write_df(df,file_path,overwrite_flag,matrix_or_table)
%--------------------------------------------------------------------------
% Save centroid list or data matrix as new file or overwrite existing file.
%--------------------------------------------------------------------------

if nargin<4 && istable(df)
    matrix_or_table = 'table';
elseif nargin<4 && isnumeric(df)
    matrix_or_table = 'matrix';
elseif ~isequal(matrix_or_table,'matrix') || ~isequal(matrix_or_table,'table')
    error("Enter whether to save as matrix or table")
end

assert(isequal(file_path(end-3:end),'.csv'),"File path must end with .csv filename")

% Check if file exists
if exist(file_path,'file') == 2
    exists = true;
else
    exists = false;
end

% If overwriting, just save file
if overwrite_flag || ~exists
   if isequal(matrix_or_table,'matrix')
       writematrix(df, file_path)
   else
        writetable(df, file_path)
   end
   return
end

% If file exists, save under new filename
if exists
    [filepath,name,ext] = fileparts(file_path);
    idx = 1;
    new_name = sprintf('%s_%d',name,idx);
    file_path_new = fullfile(filepath,sprintf('%s%s',new_name,ext));
    exists = exist(file_path_new,'file') == 2;

    % If file exists, add 1 to index
    while exists
        idx = idx+1;
        new_name = sprintf('%s_%d',name,idx);
        file_path_new = fullfile(filepath,sprintf('%s%s',new_name,ext));
        exists = exist(file_path_new,'file') == 2;
    end
    
    % Write file
    if isequal(matrix_or_table,'matrix')
       writematrix(df, file_path_new)
    else
        writetable(df, file_path_new)
    end
end

end