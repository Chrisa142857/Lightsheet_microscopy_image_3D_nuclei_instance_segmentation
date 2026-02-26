function cleanup_tmp(config)

% Remove temporary folders in output directory
folders = dir(fullfile(config.output_directory,"tmp_*"));
folders = folders([folders.isdir]);
arrayfun(@(s) rmdir(fullfile(s.folder,s.name),'s'),folders)

% Remove tmp nifti files from home directory
home_path = fileparts(which('NM_config'));
tmp_files = dir(fullfile(home_path,'tmp_reg_*'));
arrayfun(@(s) delete(fullfile(s.folder,s.name)),tmp_files)

end