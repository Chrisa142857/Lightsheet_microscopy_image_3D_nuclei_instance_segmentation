function compress_nii(path)

files = dir(path);

for i = 1:length(files)
   if isfolder(fullfile(path,files(i).name))
        subfiles = dir(fullfile(path,files(i).name));   
        for j = 1:length(subfiles)
            if ~contains(subfiles(j).name,'.nii.gz') && ...
                contains(subfiles(j).name,'.nii')
                final_path = fullfile(path,files(i).name,subfiles(j).name);
                I = niftiread(final_path);
                niftiwrite(I,final_path,'Compressed',true)
                delete(final_path)
            end
        end
   end
end

end