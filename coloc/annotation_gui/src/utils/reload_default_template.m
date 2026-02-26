function reload_default_template(stage,override)
%--------------------------------------------------------------------------
% Overwrite template files with default configuration.
%--------------------------------------------------------------------------
%
% Usage:
% reload_default_template(stage)
%
% Inputs:
% stage - NuMorph stage to load (i.e.'process', 'analyze', 'evaluate').
%--------------------------------------------------------------------------

% Prompt user what stage
if nargin<1
    stage = input("Choose which stage to load (i.e. 'process',"+...
        " 'analyze', 'evaluate'): \n");    
end

if nargin<2
    override = false;    
end

% Get location of default template
home_path = fileparts(which('NM_config'));
switch stage
    case 'process'
        file_location = fullfile(home_path,'data','defaults','NMp_template_default.m');
    case 'analyze'
        file_location = fullfile(home_path,'data','defaults','NMa_template_default.m');
    case 'evaluate'
        file_location = fullfile(home_path,'data','defaults','NMe_template_default.m');
    case 'samples'
         file_location = fullfile(home_path,'data','defaults','NM_samples_default.m');
    otherwise
        error("Unrecognized stage selected")
end

% Get confirmation from user
if ~override
    usr_confirmation = input(sprintf("Overwriting template for stage "+...
        "'%s' with default. Type 'yes' to confirm. \n",stage));  
else
    usr_confirmation = 'yes';
end
    
% Copy file
if isequal(usr_confirmation,'yes')
    [~,fname] = fileparts(file_location);
    fname = regexprep(fname,'_default','.m');
    save_location = fullfile(home_path,'templates',fname);
    [status,msg] = copyfile(file_location,save_location);
    if status 
        fprintf("Template for stage '%s' has been replaced.\n",stage)
    else
        disp(msg)
        fprintf("Error Defaults for stage '%s' has NOT been replaced.\n",stage)
    end
else
    fprintf("Template for stage '%s' has NOT been replaced.\n", stage)
end
    
    
end