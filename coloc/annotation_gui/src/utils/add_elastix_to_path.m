function add_elastix_to_path(elastix_path_bin,elastix_path_lib)

% Add elastix paths if present
if ismac || isunix
    path1 = getenv('PATH');
    if ~contains(path1,elastix_path_bin)
        path1 = [elastix_path_bin, ':', path1];
        setenv('PATH',path1)
    end

    if ismac
        path2 = getenv('DYLD_LIBRARY_PATH');
        if ~contains(path2,elastix_path_lib)
            path2 = [elastix_path_lib, ':', path2];
            setenv('DYLD_LIBRARY_PATH',path2)
        end
    else
        path2 = getenv('LD_LIBRARY_PATH');
        if ~contains(path2,elastix_path_lib)
            path2 = [elastix_path_lib, ':', path2];
            setenv('LD_LIBRARY_PATH',path2)
        end
    end
else
    % Windows is a little different
    if ispc
        if contains(elastix_path_bin,'bin')
            idx = strfind(elastix_path_bin,'\');
            elastix_path = elastix_path_bin(1:idx(end)-1);
        else
            elastix_path = elastix_path_bin;
        end
        path1 = getenv('PATH');
        if ~contains(path1,elastix_path)
            path1 = [elastix_path, ';', path1];
            setenv('PATH',path1)
        end
    end
end

end