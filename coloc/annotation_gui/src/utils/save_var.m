function save_var(config,var_name,varargin)
%--------------------------------------------------------------------------
% Save specific variable to output directory. Variable needs to be already
% present in the output directory.
%--------------------------------------------------------------------------
% Usage:
% save_var(config,variable,var_name)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Configuration structure containg output directory location.
%
% variable: Variable to save.
%
% var_name: Name of variable filename.
%
%--------------------------------------------------------------------------

if ~endsWith(var_name,'.mat')
    var_name = strcat(var_name,'.mat');
end

var_path = fullfile(config.output_directory,'variables',var_name);
    
if isfile(var_path)
    out = load(var_path,var_name);
    f_out = fields(out);
    f_in = fi
    for i = 1:length(f_out)
        varargout{i} = out.(f_out{i});
    end
else
    error("Could not locate specified variable in the output directory")
end

end
