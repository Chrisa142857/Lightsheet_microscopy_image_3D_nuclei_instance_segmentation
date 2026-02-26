function varargout=elastix(movingImage,fixedImage,outputDir,paramFile,varargin)
% elastix image registration and warping wrapper
%
% function varargout=elastix(movingImage,fixedImage,outputDir,paramFile)
%          varargout=elastix(movingImage,fixedImage,outputDir,paramFile,'PARAM1',val1,...)
% Purpose
% Wrapper for elastix image registration package. This function calls the elastix
% binary (needs to be in the system path) to conduct the registration and produce
% the transformation coefficients. These are saved to disk and, optionally, 
% returned as a variable. Transformed images are returned.
%
%
% Examples
% elastix('version')   %prints the version of elastix on your system and exits
% elastix('help')      %prints the elastix binary's help and exits
% elastix(movImage,refImage,[],'elastix_settings.yml')
% elastix(movImage,refImage,[],'elastix_settings.yml', 'paramstruct', modifierStruct)
% 
% Inputs [required]
% movingImage - A 2D or 3D matrix corresponding to a 2D image or a 3D volume. 
%               This is the image that you want to align.
%
% fixedImage -  A 2D or 3D matrix corresponding to a 2D image or a 3D volume. 
%               Must have the same number of dimensions as movingImage.
%               This is the target of the alignment. i.e. the image that you want
%               movingImage to match. 
%
% outputDir -   If empty, a temporary directory with a unique name is created 
%               and deleted once the analysis is complete. If a *valid path* is entered
%               then the directory is not deleted. If a directory is defined and it 
%               does not exist then a temporary one is created. The directory is *never deleted* if no
%               outputs are requested.
%
% paramFile - a) A string defining the name of the YAML file that contains the registration 
%             parameters. This is converted to an elastix parameter file. Leave empty to 
%             use the default file supplied with this package (elastix_default.yml)
%             Must end with ".yml"
%             b) An elastix parameter file name (relative or full path) or a cell array 
%               of elastix parameter file names. If a cell array, these are applied in 
%               order. Names must end with ".txt"
%
% 
% Inputs [optional]
% paramstruct - structure containing parameter values for the registration. This is used 
%          to modify specific parameters which may already have been defined by the .yml 
%          (paramFile). paramstruct can have a length>1, in which case these structures 
%          are treated as a request for multiple sequential registration operations. The 
%          possible values for fields in the the structure can be found in 
%          elastix_default.yml 
%          *paramstruct is ignored if paramFile is an elastix parameter file.*
%
% threads - How many threads to run the registration on. by default all available cores 
%           will be used.
% t0      - Relative or absolute path(s) (string or cell array of strings) to files 
%           defining the initial transform. If transforms are to be chained, list then 
%           in reverse order (e.g. bspline then affine).
%            
% 
% Outputs
% registered - the final registered image
% stats - all stats from the registration, including any intermediate images produced 
%         during the registration.
%
% Rob Campbell - Basel 2015
%
%
% Notes: 
% 1. You will need to download the elastix binaries (or compile the source)
% from: http://elastix.isi.uu.nl/ There are versions for all platforms. 
% 2. Not extensively tested on Windows. 
% 3. Read the elastix website and elastix_parameter_write.m to
% learn more about the parameters that can be modified. 
%
%
% Dependencies
% - elastix and transformix binaries in path
% - image processing toolbox (to run examples)


%----------------------------------------------------------------------
% *** Handle default options ***

%Confirm that the elastix binary is present
[~,elastix_version] = system('elastix --version');
r=regexp(elastix_version,'version','once');
% Modified
if isempty(r)
    fprintf('Unable to find elastix binary in system path. Quitting\n')
    return
end

%%%%% New: change outputDir to char if string
if isstring(outputDir)
    outputDir = char(outputDir);
end

%If the user supplies one input argument only and this is is a string then
%we assume it's a request for the help or version so we run it 
if nargin==1 && ischar(movingImage)
    if regexp(movingImage,'^\w')
        [~,msg]=system(['elastix --',movingImage]);
    end
    fprintf(msg)
    return
end

%%%%% New: check if images provided are cells containing multiple channels
multi_spec = false;
if iscell(movingImage)
    if length(movingImage) > 1
       multi_spec = true; 
    end
end
if iscell(fixedImage)
    if length(fixedImage) > 1
        multi_spec = true;
    end
end

if ~multi_spec && ndims(movingImage) ~= ndims(fixedImage)
    fprintf('movingImage and fixedImage must have the same number of dimensions\n')
    return
end

% Make directory into which we will write the image files and associated registration files
if nargin<3 || isempty(outputDir) 
    outputDir=fullfile(tempdir,sprintf('elastixTMP_%s_%d', datestr(now,'yymmddHHMMSS'), round(rand*1E8)));
    deleteDirOnCompletion=1;
else
    deleteDirOnCompletion=0;
end

if strcmp(outputDir(end),filesep) %Chop off any trailing fileseps 
    outputDir(end)=[];
end

if ~exist(outputDir,'dir') || isempty(outputDir)
    if ~mkdir(outputDir)
        error('Can''t make data directory %s',outputDir)
    end
end

if nargin<4
    paramFile=[];
end
if isempty(paramFile)
    defaultParam = 'elastix_default.yml';
    fprintf('Using default parameter file %s\n',defaultParam)
    paramFile = defaultParam;
end


%Handle parameter/value pairs
p = inputParser;
addOptional(p,'threads', [], @isnumeric)
addOptional(p,'t0', [])
addOptional(p,'s',[1 1 1])
addOptional(p,'verbose',0)
addOptional(p,'paramstruct', [], @isstruct)
addOptional(p,'fMask',[], @isnumeric)
addOptional(p,'mMask',[],@isnumeric)
addOptional(p,'mp',[],@isnumeric)
addOptional(p,'fp',[],@isnumeric)
parse(p,varargin{:})
threads = p.Results.threads;
t0 = p.Results.t0;
paramstruct = p.Results.paramstruct;
verbose = p.Results.verbose;
fMask = p.Results.fMask;
mMask = p.Results.mMask;
elementSpacing = p.Results.s;

%%%%% New: get points
mov_points = p.Results.mp;
fix_points = p.Results.fp;

%error check: confirm initial parameter files exist
if ~isempty(t0)
    if ischar(t0) 
       t0 = {t0}; %just to make later code neater
    end

    for ii = 1:length(t0)
        if ~exist(t0{ii},'file')
            print('Can not find initial transform %s\n', t0{ii})
            return
        end
    end    
end



%----------------------------------------------------------------------
% *** Conduct the registration ***

if strcmp('.',outputDir)
    [~,dirName]=fileparts(pwd);
else
    [~,dirName]=fileparts(outputDir);
end


% Create and move the images

if ~iscell(movingImage)
    movingFname=sprintf('%s_moving',dirName); %TODO: so the file name contains the dir name?
    mhd_write(movingImage,movingFname,elementSpacing);
    if ~strcmp(outputDir,'.')
        if ~movefile([movingFname,'.*'],outputDir); error('Can''t move files'), end
    end
else
    for i = 1:length(movingImage)
        movingFname(i,:)= sprintf('%s_moving_%d',dirName,i); %TODO: so the file name contains the dir name?
        movingSubImage = movingImage{i};
        mhd_write(movingSubImage,movingFname(i,:),elementSpacing);
        if ~strcmp(outputDir,'.')
            if ~movefile([movingFname(i,:),'.*'],outputDir); error('Can''t move files'), end
        end
    end
end

if ~iscell(fixedImage)
    fixedFname=sprintf('%s_target',dirName);
    mhd_write(fixedImage,fixedFname,elementSpacing);
    if ~strcmp(outputDir,'.') %Don't copy if we're already in the directory
        if ~movefile([fixedFname,'.*'],outputDir); error('Can''t move files'), end
    end
else
    for i = 1:length(fixedImage)
        fixedFname(i,:)= sprintf('%s_target_%d',dirName,i); %TODO: so the file name contains the dir name?
        fixedSubImage = fixedImage{i};
        mhd_write(fixedSubImage,fixedFname(i,:),elementSpacing);
        if ~strcmp(outputDir,'.')
            if ~movefile([fixedFname(i,:),'.*'],outputDir); error('Can''t move files'), end
        end
    end
end


%%%%%New 
%Write fixed mask image if provided
if ~isempty(fMask)
    fMask = uint8(fMask);
    maskFname = [dirName, '_fMask'];
    mhd_write(fMask,maskFname,elementSpacing);
    if ~strcmp(outputDir,'.')
        if ~movefile([maskFname,'.*'],outputDir); error('Can''t move files'), end
    end
end

%%%%% New: Write moving mask image if provided
if ~isempty(mMask)
    mMask = uint8(mMask);
    maskMname = [dirName, '_mMask'];
    mhd_write(mMask,maskMname,elementSpacing);
    if ~strcmp(outputDir,'.')
        if ~movefile([maskMname,'.*'],outputDir); error('Can''t move files'), end
    end
end


%%%%% New: Write points files if provided
if ~isempty(mov_points)
    mov_point_Fname=fullfile(outputDir,'tmp_mov_pts.txt');
    writePointsFile(mov_point_Fname,mov_points)
end

if ~isempty(fix_points)
    fix_point_Fname=fullfile(outputDir,'tmp_fix_pts.txt');
    writePointsFile(fix_point_Fname,fix_points)
end




%Build the parameter file(s)
if ischar(paramFile) && strfind(paramFile,'.yml') && ~isempty(paramstruct) %modify settings from YAML with paramstruct
    for ii=1:length(paramstruct)
        paramFname{ii}=sprintf('%s_parameters_%d.txt',dirName,ii);
        paramFname{ii}=fullfile(outputDir,paramFname{ii});
        elastix_parameter_write(paramFname{ii},paramFile,paramstruct(ii))
    end

elseif ischar(paramFile) && strfind(paramFile,'.yml') && isempty(paramstruct) %read YAML with no modifications
    paramFname{1} = fullfile(outputDir,sprintf('%s_parameters_%d.txt',dirName,1));
    elastix_parameter_write(paramFname{1},paramFile)

elseif (ischar(paramFile) && strfind(paramFile,'.txt')) %we have an elastix parameter file
    if ~strcmp(outputDir,'.')
        copyfile(paramFname,outputDir)
        paramFname{1} = fullfile(outputDir,paramFname);
    end

elseif iscell(paramFile) %we have a cell array of elastix parameter files
    paramFname = paramFile;
     if ~strcmp(outputDir,'.') 
        for ii=1:length(paramFname)
            if ~isfile(paramFname{ii})
                error("Parameter filename path does not exist")
            end
            copyfile(paramFname{ii},outputDir)
            %So paramFname is now:
            [~,f,e] = fileparts(paramFname{ii});
            paramFname{ii} = fullfile(outputDir,[f,e]);
        end
    end

else
    error('paramFile format in file not understood')    
end


%If the user asked for an initial transform, collate the transform files, copy them to the 
%transform directory, and ensure they are linked.
if ~isempty(t0)
    copiedLocations = {}; %Keep track of the locations to which the files are stored
    for ii=1:length(t0)
        [fPath,pName,pExtension] = fileparts(t0{ii});
        copiedLocations{ii} = fullfile(outputDir,['init_',pName,pExtension]);
        if verbose
            fprintf('Copying %s to %s\n',t0{ii},copiedLocations{ii})
        end
        copyfile(t0{ii},copiedLocations{ii})
    end

    %Modify the parameter files so that they chain together correctly
    for ii=1:length(t0)-1
        changeParameterInElastixFile(copiedLocations{ii},'InitialTransformParametersFileName',copiedLocations{ii+1},verbose)
    end

    %Add the first parameter file to the command string 
    initCMD = sprintf(' -t0 %s ',copiedLocations{1});
else
    initCMD = '';
end


%%%% New: add mask, points files to command
%If fixed mask is provided, add to command string
if ~isempty(fMask)
    maskLocation = fullfile(outputDir,sprintf('%s.mhd',maskFname));
    initCMD = sprintf('%s -fMask %s ',initCMD, maskLocation);
end

%If moving mask is provided, add to command string
if ~isempty(mMask)
    maskLocation = fullfile(outputDir,sprintf('%s.mhd',maskMname));
    initCMD = sprintf('%s -mMask %s ',initCMD, maskLocation);
end


%If points
if ~isempty(mov_points)
    initCMD = sprintf('%s -mp %s ',initCMD, mov_point_Fname);
end

%If moving mask is provided, add to command string
if ~isempty(fix_points)
    initCMD = sprintf('%s -fp %s ',initCMD, fix_point_Fname);
end


%%%%% New: build seperate commands for single or multi-channel registration
if ~multi_spec
    CMD=sprintf('elastix -f %s.mhd -m %s.mhd -out %s ',...
                fullfile(outputDir,fixedFname),...
                fullfile(outputDir,movingFname),...
                outputDir);
else
    movingFnames = "";
    for i = 1:size(movingFname,1)
       movingFnames = movingFnames + ...
           sprintf("-m%d %s ",i-1, fullfile(outputDir, strcat(string(movingFname(i,:)),'.mhd')));
    end
    fixedFnames = "";
    for i = 1:size(fixedFname,1)
       fixedFnames = fixedFnames + ...
           sprintf("-f%d %s ",i-1, fullfile(outputDir, strcat(string(fixedFname(i,:)),'.mhd')));
    end
    
    CMD=sprintf('elastix %s %s -out %s ',fixedFnames,movingFnames,outputDir);
end
CMD = [CMD,initCMD];

if ~isempty(threads)
    CMD = sprintf('%s -threads  %d',CMD,threads);
end

%Loop through, adding each parameter file in turn to the string
for ii=1:length(paramFname) 
    CMD=[CMD,sprintf(' -p %s ', paramFname{ii})];
end

%store a copy of the command to the directory
cmdFid = fopen(fullfile(outputDir,'CMD'),'w');
fprintf(cmdFid,'%s\n',CMD);
fclose(cmdFid);

% Run the command and report back if it failed
fprintf('Running: %s\n',CMD)
if isunix
    %CMD=['LD_LIBRARY_PATH= ', CMD];
end

[status,result]=system(CMD);

if status %Things failed. Oh dear. 
    if status
        fprintf('\n\t*** Transform Failed! ***\n%s\n',result)
    else
        disp(result)
    end
    registered=[];
    out.outputDir=outputDir;
    out.TransformParameters=nan;
    out.TransformParametersFname=nan;    

    if deleteDirOnCompletion
        fprintf('Keeping temporary directory %s for debugging purposes\n',outputDir)
    end


    % Modified: return only transform parameters if nargout==1. If
    % nargout>1, return also the transformed image as seperate variable
else %Things worked! So let's return stuff to the user 
    
    %Return the transform parameters
    d=dir(fullfile(outputDir,'TransformParameters.*.txt'));
    for ii=1:length(d)
        out.TransformParameters{ii}=elastix_parameter_read([outputDir,filesep,d(ii).name]);
        out.TransformParametersFname{ii}=[outputDir,filesep,d(ii).name];
    end

    out.log=readWholeTextFile([outputDir,filesep,'elastix.log']);
    out.outputDir=outputDir; %may be a relative path
    out.currentDir=pwd;
    out.movingFname=movingFname;
    out.targetFname=fixedFname;

    if nargout>1      
        %return the final transformed image
        d=dir(fullfile(outputDir,'result*.mhd'));
        if isempty(d)
            fprintf('WARNING: could find no transformed result images in %s\n',outputDir);
            registered=[];
        else
            registered=mhd_read([outputDir,filesep,d(end).name]);
        end
    end
end


%Optionally return to the command line
if nargout>0
    varargout{1}=out;
end

if nargout>1
    varargout{2}=registered;
    if deleteDirOnCompletion
        fprintf('Deleting temporary directory %s\n',outputDir)
        rmdir(outputDir,'s')
    end
    varargout{3} = status;
end
