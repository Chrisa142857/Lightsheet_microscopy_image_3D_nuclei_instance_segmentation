function disp_counts(config,index,bin)
%--------------------------------------------------------------------------
% Display total cell counts for particular annotation index. 
%--------------------------------------------------------------------------
% Usage:
% disp_counts(config,index,bin)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Configuration structure containg output directory location.
%
% index: (numeric or string) Structure index to query. Can also provide
% structure acronym as string if registered to atlas.
% 
% bin: (logical) Bin counts along structure tree. (default: true)
% 
%--------------------------------------------------------------------------

if nargin<2
    error("Specify structure index")
end

if nargin<3
    bin = true;
end

home_path = fileparts(which('NM_config'));
c = load_results(config,'summary');

temp = readtable(fullfile(home_path,'annotations','structure_template.csv'));

if isstring(index) || ischar(index)
    index = temp.index(string(temp.acronym) == string(index));
end

if bin
    id = temp.id(temp.index == index);
    idx = cellfun(@(s) contains(s,string(id)),temp.structure_id_path);
    index = temp.index(idx);
    count = sum(c.counts(ismember(c.counts(:,1),index),2:end),'all');
else
    count = sum(c.counts(c.counts(:,1) == index,2:end),'all');
end

disp(count)

end