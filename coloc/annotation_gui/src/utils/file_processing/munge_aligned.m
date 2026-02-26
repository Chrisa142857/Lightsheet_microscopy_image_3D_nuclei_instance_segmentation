function path_table_series = munge_aligned(config)
%From alignment step  

% Remove unnecessary fields
fields_to_remove = {'folder','date','isdir','bytes','datenum'};

% Get directory contents
paths_sub = dir(fullfile(config.output_directory,'aligned'));

% Check .tif in current folder
paths_sub = paths_sub(arrayfun(@(x) contains(x.name,'.tif'),paths_sub));

% Create new field for file location
C = arrayfun(@(x) fullfile(paths_sub(1).folder,x.name),paths_sub,'UniformOutput',false);
[paths_sub.file] = C{:};

paths_sub = rmfield(paths_sub,fields_to_remove);

% Generate table components
components = arrayfun(@(s) strsplit(s.name,{char(config.sample_id),'_','.'}), paths_sub, 'UniformOutput', false);
components = vertcat(components{:});

assert(size(components,1) > 0, "Could not recognize any aligned files by filename structure")
assert(length(unique(components(:,1))) == 1, "Multiple sample ids found in image path")

%Take image information
for i = 1:length(paths_sub)
    paths_sub(i).sample_id = string(config.sample_id);
    paths_sub(i).markers = string(components{i,4});
end

positions = cellfun(@(s) str2double(s),components(:,[6,5,2]),'UniformOutput',false);
channel_num = cellfun(@(s) str2double(s(2)),components(:,3),'UniformOutput',false);
[paths_sub.channel_num] = channel_num{:,1};

[paths_sub.y] = positions{:,2};
[paths_sub.x] = positions{:,1};
[paths_sub.z] = positions{:,3};

path_table_series = struct2table(rmfield(paths_sub,{'name'}));


end