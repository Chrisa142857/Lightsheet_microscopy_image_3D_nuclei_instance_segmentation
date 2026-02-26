function [l_idx,index] = bin_annotation_layers(input)
%-------------------------------------------------------------------------
% This function bin annotations by their cortical layer. Input can be a
% mask or vector of indexes
%-------------------------------------------------------------------------

% Create new array for storing indexes
l_idx = zeros(size(input), class(input));
index = 0;
home_path = fileparts(which('NM_config'));

% Read layer specific tables and replace indexes
l_table = readtable(fullfile(home_path,'annotations','default_annotations','cortical_layers','layer1.csv'));
idx = ismember(input,l_table.index);
if ~isempty(idx)
    l_idx(idx) = 1;
    index = horzcat(index,1);
end

l_table = readtable(fullfile(home_path,'annotations','default_annotations','cortical_layers','layer2_3.csv'));
idx = ismember(input,l_table.index);
if ~isempty(idx)
    l_idx(idx) = 23;
    index = horzcat(index,23);
end

l_table = readtable(fullfile(home_path,'annotations','default_annotations','cortical_layers','layer4.csv'));
idx = ismember(input,l_table.index);
if ~isempty(idx)
    l_idx(idx) = 4;
    index = horzcat(index,4);
end

l_table = readtable(fullfile(home_path,'annotations','default_annotations','cortical_layers','layer5.csv'));
idx = ismember(input,l_table.index);
if ~isempty(idx)
    l_idx(idx) = 5;
    index = horzcat(index,5);
end

l_table = readtable(fullfile(home_path,'annotations','default_annotations','cortical_layers','layer6a.csv'));
idx = ismember(input,l_table.index);
if ~isempty(idx)
    l_idx(idx) = 60;
    index = horzcat(index,60);
end

l_table = readtable(fullfile(home_path,'annotations','default_annotations','cortical_layers','layer6b.csv'));
idx = ismember(input,l_table.index);
if ~isempty(idx)
    l_idx(idx) = 61;
    index = horzcat(index,61);
end
index = index(index>0);

end


