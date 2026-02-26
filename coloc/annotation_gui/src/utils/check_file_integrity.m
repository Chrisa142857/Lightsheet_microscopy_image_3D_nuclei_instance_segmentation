function error_files = check_file_integrity(input_directory, chunk_size)
%--------------------------------------------------------------------------
% Check for corrupt images in directories by opening .tiffs using MATALB's
% imread function.
%--------------------------------------------------------------------------

if nargin<1
    error("Missing input image file directory")
end

if nargin<2
    chunk_size = 10;
end

dir_files = dir(input_directory);

% Find all tif files in directory and subdirectory
file_list = cell(1,length(dir_files));
for i = 1:length(dir_files)
   if isequal(dir_files(i).name,'.') || isequal(dir_files(i).name,'..')
       continue
   end
   
   if contains(dir_files(i).name,{'.tif','.tiff'})
       file_list{i} = {fullfile(input_directory,dir_files(i).name)};
       continue
   end
   
   if isfolder(fullfile(input_directory,dir_files(i).name))
       sub = dir(fullfile(input_directory,dir_files(i).name));
       sub = sub(arrayfun(@(s) contains(s.name,{'.tif','.tiff'}),sub),:);
       if ~isempty(sub)
            file_list{i} = fullfile(sub(1).folder,{sub.name});
       end
   end
end

% Combine files
file_list = [file_list{:}];

% Check files using imread
error_files = {};
error_detect = false;
fprintf("Checking %d files in directory %s\n",length(file_list),input_directory)
for i = 1:length(file_list)
    try
        img = imread(file_list{i},'PixelRegion',{[1,chunk_size],[1,chunk_size]});
    catch
        fprintf("Error reading image %s\n",file_list{i})
        error_files = cat(1,error_files,file_list(i));
        error_detect = true;
    end    
end

% Print if no errors detected
if ~error_detect
    fprintf("No errors detected! \n")
end

end

