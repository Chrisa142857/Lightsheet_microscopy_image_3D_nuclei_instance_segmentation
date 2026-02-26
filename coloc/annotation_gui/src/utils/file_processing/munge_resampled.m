function path_table_nii = munge_resampled(config)
% From resampled step

% Remove unnecessary fields
fields_to_remove = {'folder','date','isdir','bytes','datenum'};

% Get directory contents
paths_sub = dir(fullfile(config.output_directory,'resampled'));
    
% Check .nii in current folder
paths_sub = paths_sub(arrayfun(@(x) contains(x.name,'.nii'),paths_sub));

% Create new field for file location
C = arrayfun(@(x) fullfile(paths_sub(1).folder,x.name),paths_sub,'UniformOutput',false);
if length(C) > 1
    [paths_sub.file] = C{:};
else
    [paths_sub.file] = C(:);
end
paths_sub = rmfield(paths_sub,fields_to_remove);

% Generate table components
components = arrayfun(@(s) strsplit(s.name,{'_','.'}), paths_sub, 'UniformOutput', false);
components = vertcat(components{:});

% Take image information
for i = 1:length(paths_sub)
    paths_sub(i).sample_id = string(components{i,1});
    paths_sub(i).markers = string(components{i,3});
end

channel_num = cellfun(@(s) str2double(s(2)),components(:,2),'UniformOutput',false);
[paths_sub.channel_num] = channel_num{:,1};

% Set x,y tiles as 1 and save z positions
positions = cellfun(@(s) str2double(s),components(:,4),'UniformOutput',false);

[paths_sub.y_res] = positions{:,1};
[paths_sub.x_res] = positions{:,1};
[paths_sub.z_res] = positions{:,1};

path_table_nii = struct2table(rmfield(paths_sub,{'name'}));

end