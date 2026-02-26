function list_samples
%--------------------------------------------------------------------------
% List all samples in NM_samples.
%--------------------------------------------------------------------------
% Usage:
% list_samples
%
%--------------------------------------------------------------------------

home_path = fileparts(which('NM_config'));
fid = fopen(fullfile(home_path,'templates','NM_samples.m'));
c = textscan(fid,'%s','Delimiter','\n');
fclose(fid);

c = c{:};
c = extractAfter(c,'case ');
c = cellfun(@(s) string(s(2:end-1)),c);

fprintf("'%s'\n",c(c ~= ""))

end