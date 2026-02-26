function conda_path = add_conda_to_path(conda_path)

if nargin<1
    conda_path = [];
end

% First check if conda binary already in PATH
PATH = getenv('PATH');
if contains(PATH,'conda')
    a = strsplit(PATH,':');
    idx = cellfun(@(s) contains(s,'conda'),a);    
    conda_path = a{find(idx,1)};
    return
end

% Scan for location of conda binary
if  ismac || isunix       
    userDir = char(java.lang.System.getProperty('user.home'));
    locations = {'.bash_profile','.bashrc','.zshrc','.zprofile'};
    
    for i = 1:length(locations)
        if isfile(fullfile(userDir,locations{i}))
            fid = fopen(fullfile(userDir,locations{i}));
            c = textscan(fid,'%s');
            fclose(fid);
            
            if any(contains(c{:},{'conda3/bin:$PATH'}))
                idx = find(contains(c{:},{'conda3/bin:$PATH'}),1);
                s1 = regexp(c{1}{idx,:},'/');
                s2 = regexp(c{1}{idx,:},':');
                
                conda_path = c{1}{idx,:}(s1(1):s2(1)-1);
            end
        end
    end
else
    % Get conda path from NM_config if not provided
    if nargin<1
        filetext = fileread('NM_config.m');
        expr = '[^\n]*conda_path[^\n]*';
        matches = regexp(filetext,expr,'match');
        conda_match = strsplit(matches{1},{''''});
        conda_match = conda_match{2};
    else
        conda_match = conda_path;
    end

    if isequal(conda_match,'default')
        warning("For Windows OS, specify full path to conda folder as conda_path "+...
            "in NM_config")
        pause(5)
        return
    else
        conda_path = strcat(conda_match,'\Library\bin');
        conda_path = strcat(conda_path,';',conda_match);
        conda_path = strcat(conda_path,';',conda_match,'\Library\Scripts');
    end
end

% Add to PATH
if ~isempty(conda_path)
    PATH = conda_path + ":" + PATH; 
    setenv('PATH',PATH)
end

end