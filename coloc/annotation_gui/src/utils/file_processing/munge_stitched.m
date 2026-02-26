function path_table_series = munge_stitched(config)
%From stitching step  

% Remove unnecessary fields
fields_to_remove = {'folder','date','isdir','bytes','datenum'};

% Get directory contents
paths_sub = dir(fullfile(config.output_directory,'stitched'));

% Check .tif in current folder
paths_sub = paths_sub(arrayfun(@(x) contains(x.name,'.tif'),paths_sub));

% Check if any files present
if isempty(paths_sub)
    error("No proper images found in stitched directory")
end

% Create new field for file location
C = arrayfun(@(x) fullfile(paths_sub(1).folder,x.name),paths_sub,'UniformOutput',false);
[paths_sub.file] = C{:};

% Remove unused fields
paths_sub = rmfield(paths_sub,fields_to_remove);

% Generate table components
components = arrayfun(@(s) strsplit(s.name,{'_','.'}), paths_sub, 'UniformOutput', false);
components = vertcat(components{:});

% Take image information
for i = 1:length(paths_sub)
    paths_sub(i).sample_id = string(components{i,1});
    paths_sub(i).markers = string(components{i,4});
end

channel_num = cellfun(@(s) str2double(s(2)),components(:,3),'UniformOutput',false);
[paths_sub.channel_num] = channel_num{:,1};

% Set x,y tiles as 1 and save z positions
positions = cellfun(@(s) str2double(s),components(:,[6,5,2]),'UniformOutput',false);
x = num2cell(ones(1,length(paths_sub)));

[paths_sub.x] = x{:};
[paths_sub.y] = x{:};
[paths_sub.z] = positions{:,3};

path_table_series = struct2table(rmfield(paths_sub,{'name'}));

end