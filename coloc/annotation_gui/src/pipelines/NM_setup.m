function NM_setup(option)
% Download and verify MATLAB add-ons and external packages used in NuMorph
if nargin < 1; option = 'notlight'; end
if isequal(option, 'light'); islight = true; else; islight = false; end

o = weboptions('CertificateFilename','');
home_path = fileparts(which('NM_config'));

% Create empty directories
tmp_folder = fullfile(home_path,'data','tmp');
if ~isfolder(tmp_folder)
    mkdir(tmp_folder)
end

% Check Matlab version and required add-ons
fprintf("Checking MATLAB toolboxes \n")
v = ver;
% MATLAB version
idx = v(arrayfun(@(s) isequal(s.Name,'MATLAB'),v)).Release;
idx = str2double(idx([5,6]));
if idx < 20
    warning("Detected MATLAB release %s. Some functions may not exist or work correctly"+...
        "in this release. Update MATLAB to at least version R2020a.",v(1).Release)
    pause(5)
end

names = arrayfun(@(s) string(s.Name),v);
    
% Image processing toolbox
if ~any(ismember(names,"Image Processing Toolbox"))
    warning("Image Processing Toolbox does not exist. "+...
        "Please install this toolbox through the MATLAB Add-Ons menu.")
    pause(5)
end
    
% Statistics and machine learning toolbox
if ~any(ismember(names,"Statistics and Machine Learning Toolbox"))
    warning("Statistics and Machine Learning Toolbox does not exist. "+...
        "Please install this toolbox through the MATLAB Add-Ons menu.")
    pause(5)
end

% Parallel computing toolbox
if ~any(ismember(names,"Parallel Computing Toolbox"))
    warning("Parallel Computing Toolbox does not exist. "+...
        "Please install this toolbox through the MATLAB Add-Ons menu.")
    pause(5)
end

% Check for reference atlases
fprintf("Checking for reference atlases \n")
atlas_path1 = fullfile(home_path,'data','atlas','average_template_25.nii');
atlas_path2 = fullfile(home_path,'data','atlas','ara_nissl_25.nii');
if ~isfile(atlas_path1) && ~isfile(atlas_path2)
    if ~isfile(atlas_path1)
       fprintf("Downloading average template atlas...\n")
       websave(atlas_path1,"https://bitbucket.org/steinlabunc/numorph/downloads/average_template_25.nii",o);
    end
    if ~isfile(atlas_path2)
       fprintf("Downloading ara nissl atlas...\n")
       websave(atlas_path2,"https://bitbucket.org/steinlabunc/numorph/downloads/ara_nissl_25.nii",o);
    end
else
    fprintf("All atlases exist \n\n")
end

% Check for annotations
flatview_path = fullfile(home_path, 'data', 'annotation_data', 'flatviewCortex.mat');
olf_cer_path = fullfile(home_path, 'data', 'annotation_data', 'olf_cer.mat');
ccfv3_path = fullfile(home_path, 'data', 'annotation_data', 'ccfv3.mat');
if ~isfile(flatview_path) && ~isfile(olf_cer_path) && ~isfile(ccfv3_path)
   fprintf("Downloading annotation data...\n")
    if ~isfile(ccfv3_path)
       websave(ccfv3_path,"https://bitbucket.org/steinlabunc/numorph/downloads/ccfv3.mat",o);
    end
    if ~isfile(olf_cer_path)
       websave(olf_cer_path,"https://bitbucket.org/steinlabunc/numorph/downloads/olf_cer.mat",o);
    end
    if ~isfile(flatview_path)
       websave(flatview_path,"https://bitbucket.org/steinlabunc/numorph/downloads/flatviewCortex.mat",o);
    end
else
    fprintf("All annotation data exist \n\n")
end

% Move templates 
template_path = fullfile(home_path,'templates');
if ~isfolder(template_path)
    mkdir(template_path)
    reload_default_template('process',true)
    reload_default_template('analyze',true)
    reload_default_template('evaluate',true)
    reload_default_template('samples',true)
else
    if ~isfile(fullfile(template_path, 'NMp_template.m'))
        reload_default_template('process',true)
    end
    if ~isfile(fullfile(template_path, 'NMa_template.m'))
        reload_default_template('analyze',true)
    end
    if ~isfile(fullfile(template_path, 'NMe_template.m'))
        reload_default_template('evaluate',true)
    end
    if ~isfile(fullfile(template_path, 'NM_samples.m'))
        reload_default_template('samples',true)
    end
end
addpath(template_path)

% Return if only light installation
if islight; return; end

% Check for elastix
fprintf("Checking for elastix \n")
external_path = fullfile(home_path,'src','external');
elastix_path = fullfile(external_path,'elastix');

% Note, some library issues found while calling elastix-5.0 from linux os.
% Downloading elastix-4.9 for this one instead since all main registration
% functions should be the same
if ~isfolder(elastix_path)
   mkdir(elastix_path)
    if ismac
       fprintf("Downloading elastix-5.0.0 for mac...\n")
        out = websave(fullfile(elastix_path,"elastix-5.0.0-mac.tar.gz"),...
            "https://github.com/SuperElastix/elastix/releases/download/5.0.0/elastix-5.0.0-mac.tar.gz",o);
        untar(out,elastix_path)
        delete(out)
    elseif isunix
       fprintf("Downloading elastix-4.9.0 for linux...\n")
        out = websave(fullfile(elastix_path,"elastix-4.9.0-linux.tar.bz2"),...
            "https://github.com/SuperElastix/elastix/releases/download/4.9.0/elastix-4.9.0-linux.tar.bz2",o);
        cmd = sprintf("tar -xf %s -C %s", out,elastix_path);
        system(cmd);
        delete(out)
    elseif ispc
       fprintf("Downloading elastix-5.0.0 for windows...\n")
        out = websave(fullfile(elastix_path,"elastix-5.0.0-win64.zip"),...
            "https://github.com/SuperElastix/elastix/releases/download/5.0.0/elastix-5.0.0-win64.zip",o);
        unzip(out,elastix_path)
        delete(out)
    end
else
    fprintf("Elastix binary already exists in /src/external \n")
end

% Add to PATH and verify that elastix can be called from matlab
elastix_path_bin = fullfile(home_path,'src','external','elastix','bin');
elastix_path_lib = fullfile(home_path,'src','external','elastix','lib');
add_elastix_to_path(elastix_path_bin,elastix_path_lib)

[c,elastix_version] = system('elastix --version');
if c>0
    warning("Errors calling elastix binary from MATLAB")
    disp(elastix_version)
    pause(5)
else
    fprintf("Successfully called %s \n",elastix_version)
end

%  Check if vl_feat toolbox exists
fprintf("Checking for vl_feat toolbox \n")
vl_path = fullfile(external_path,'vlfeat-0.9.21');
if ~isfolder(vl_path)
   mkdir(vl_path)
   fprintf("Downloading vl_feat toolbox...\n\n")
   out = websave(fullfile(vl_path,"vlfeat-0.9.21-bin.tar.gz"),...
       "https://www.vlfeat.org/download/vlfeat-0.9.21-bin.tar.gz",o);
   untar(out,vl_path)
   delete(out)
   %addpath(genpath(external_path))
else
    fprintf("vl_feat toolbox already exists already exists in /src/external \n\n")
end

% Setup enviornment for 3d-unet
% Check if conda is installed
PATH = getenv('PATH');
c_idx = 1;
if ~contains(PATH,'conda/bin') && ~contains(PATH,'conda3/bin')
    conda_path = add_conda_to_path;
    if isempty(conda_path)
        warning("Did not detect conda installation. Download and install "+...
            "miniconda for python version 3. If you're sure that conda is installed and is callable "+...
            "from shell, update conda binary location in NM_config")
        pause(5)
        c_idx = 0;
    end
end

% Check if 3dunet-centroid environment is installed
if c_idx ~= 0 
    fprintf("Checking for 3D-Unet conda environment \n")
    [~,envs] = system('conda env list');
    if ~contains(envs,'3dunet-centroid')
        fprintf("Creating conda environment to run centroid prediction using 3D-Unet \n")
        if ismac
            env_path = fullfile(home_path,'src','analysis','3dunet','environment_mac.yml');
        elseif isunix
            env_path = fullfile(home_path,'src','analysis','3dunet','environment_unix.yml');
        elseif ispc
            env_path = fullfile(home_path,'src','analysis','3dunet','environment_pc.yml');
        end
        CMD = sprintf("conda env create -f %s",env_path);
        status = system(CMD);
        if status == 0 
            fprintf("Conda environemnt successfully installed \n")
        else
            warning("Errors occured during conda installation \n")
            pause(5)
        end 
    else
        fprintf("Conda environment already installed \n")
    end
end

% Check if unet model exists
if c_idx ~= 0 
    fprintf("Checking for 3D-Unet model files \n")
    model_files = dir(fullfile(home_path,'src','analysis','3dunet','nuclei','models'));
    if ~any(endsWith({model_files.name},'.h5'))
       fprintf("Downloading 3D-Unet model 075_121_model.h5...\n")
        out = websave(fullfile(model_files(1).folder,'075_121_model.h5'),...
            "https://bitbucket.org/steinlabunc/numorph/downloads/075_121_model.h5",o);
    else
        fprintf("Model file %s already exists \n",...
            model_files(arrayfun(@(s) endsWith(s.name,'.h5'),model_files)).name)
    end
end

fprintf("Setup completed! \n")

end