function res_struct = munge_stats(var_path)
%--------------------------------------------------------------------------
% Convert results stats table to organized structure
%--------------------------------------------------------------------------
% Usage:
% res_struct = munge_stats(var_path)
%
%--------------------------------------------------------------------------
% Inputs:
% var_path: (string) Path to stats .xls table (e.g. labeled "_stats.xls").
%
%--------------------------------------------------------------------------
% Outputs:
% res_struct: Results structure.
%
%--------------------------------------------------------------------------

var_path = string(var_path);

% Read through possible sheets
res_tbl = cell(1,4);
sheetname = {'Volume','Counts','Density','Cortex'};
for i = 1:length(res_tbl)
    try 
        tbl = readtable(var_path,'PreserveVariableNames',true,'Sheet',sheetname{i});
        var_names = tbl.Properties.VariableNames;
        
        if i == 1
            idx = find(contains(var_names,'Volume'),1);
        elseif i == 2
            idx = find(contains(var_names,'Counts'),1);
        elseif i == 3
            idx = find(contains(var_names,'Density'),1);
        elseif i == 4
            idx = find(contains(var_names,'volume'),1);
        end
        res_struct.(sheetname{i}).info = tbl(:,1:idx-1);
        m = table2array(tbl(:,idx:end));
        
        if i>1
            a = cellfun(@(s) strsplit(s,'_'),var_names(idx:end),'UniformOutput',false);
            if i ~=4
                a = unique(cellfun(@(s) s(end-1),a),'stable');
            else
                a = unique(cellfun(@(s) s(end),a),'stable');
            end
        else            
            res_struct.(sheetname{i}).stats.marker = "Volume";
            res_struct.(sheetname{i}).stats.safe_name = "Volume";
            res_struct.Volume.stats.stats = m;
            continue
        end
        
        b = 1;
        for j = 1:length(a)
            res_struct.(sheetname{i}).stats(j).marker = string(a{j});
            res_struct.(sheetname{i}).stats(j).safe_name = strrep(string(a{j}),'./','');
            res_struct.(sheetname{i}).stats(j).stats = m(:,b:b+7);
            b = b+8;
        end
    catch
        if i == 1
            error("Could not read stats table");
        end
    end
end


end