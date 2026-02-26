function [config, path_table] = perform_channel_alignment(config, path_table, equal_res)
%--------------------------------------------------------------------------
% Align multiple channels for a given raw image set.
%--------------------------------------------------------------------------
% Usage:
% [config, path_table] = perform_channel_alignment(config, path_table, 
% equal_res)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Process configuration structure.
%
% path_table: Path table containing image information.
%
% equal_res: (logical) Whether image channels are all the same resolution.
% (This for testing; not currently functional)
%
%--------------------------------------------------------------------------
% Outputs:
% config: Updated configuration structure.
%
% path_table: Updated path table.
%
%--------------------------------------------------------------------------

% Check selection
if isequal(config.channel_alignment,"false")
    fprintf("%s\t No channel alignment selected. \n",datetime('now'));
    return
elseif ~isequal(config.channel_alignment, "true") && ~isequal(config.channel_alignment, "update")
    error("Unrecognized selection for channel_alignment. "+...
    "Valid options are ""true"", ""update"", or ""false"".")
end

% Count number of x,y tiles
ncols = length(unique(path_table.x));
nrows = length(unique(path_table.y));
nb_tiles = ncols * nrows;
position_mat = reshape(1:nb_tiles,[ncols,nrows])';

% Check if z resolution is consistent for all channels
if ~equal_res
    error("NuMorph currently does not support multi-channel alignment at "+...
            "multiple z resolutions")
end

if isequal(config.align_method,'translation')
    fprintf("%s\t Aligning channels by translation \n",datetime('now'));
    
    % Subset channels to align
    if isempty(config.align_channels)
        config.align_channels = 2:length(config.markers);
    end
    align_markers = config.markers(config.align_channels);
    
    % Load alignment table if it exists
    save_path = fullfile(config.output_directory,'variables','alignment_table.mat');
    if isfile(save_path)
        fprintf("%s\t Loading alignment table that already exists \n",datetime('now'));
        load(save_path,'alignment_table')
        % Check rows and columns
        assert(all(size(alignment_table) == [nrows,ncols]), "Loaded alignment table is not the same shape "+...
            "as the tiles in the input image directory")
    else
        % Create variable
        alignment_table = cell(nrows,ncols);
    end
    
    % Check for intensity thresholds which are required for alignment
    if isempty(config.lowerThresh) || isempty(config.upperThresh) || isempty(config.signalThresh)
        config = check_for_thresholds(config,path_table);
    end

    % Determine z displacement from reference channel
    % Check first if .mat file exists in output directory
    z_align_file = fullfile(config.output_directory,'variables','z_displacement_align.mat');
    if isfile(z_align_file)
        fprintf("%s\t Loading z displacement matrix \n",datetime('now'));
        load(z_align_file,'z_displacement_align')
        
        % Check that z_displacement calculated for each marker to align
        idx = ~ismember(convertStringsToChars(align_markers),fields(z_displacement_align'));        
        if any(idx)
            z_disp_channels = find(idx);
            warning("Missing z displacement for marker %s.",align_markers(z_disp_channels))
        else
            z_disp_channels = [];
        end
    else
        z_disp_channels = config.align_channels;
        z_displacement_align = [];
    end

    % Perform z adjustment calculation
    if ~isempty(z_disp_channels) || isequal(config.update_z_adjustment,"true")    
        for k = 1:length(config.align_channels)
            align_marker = config.markers(config.align_channels(k));
            fprintf("%s\t Performing channel alignment z calculation for %s/%s \n",...
                datetime('now'),align_marker,config.markers(1));
            z_tile = zeros(1,numel(position_mat));
            ave_signal = zeros(1,numel(position_mat));
            for idx = 1:numel(position_mat)
                [y,x] = find(position_mat==idx);
                path_ref = path_table(path_table.x==x & path_table.y==y & path_table.markers == config.markers(1),:);
                path_mov = path_table(path_table.x==x & path_table.y==y & path_table.markers == align_marker,:);
                % This measures the displacement in z for a given channel to the reference
                [z_tile(idx),ave_signal(idx)] = z_align_channel(config,path_mov,path_ref,config.align_channels(k));
                fprintf("%s\t Predicted z displacement of %d for tile %d x %d \n",...
                    datetime('now'),z_tile(idx),y,x);
            end
            z_matrix = reshape(z_tile,[ncols, nrows])';
            % Check for outliers
            z_displacement_align.(align_markers(k)) = z_matrix;
        end
        % Save displacement variable to output directory
        fprintf("%s\t Saving z displacement matrix \n",datetime('now'));
        save(fullfile(config.output_directory,'variables','z_displacement_align.mat'), 'z_displacement_align')
    end

    % Define which tiles to align if only doing subset
    tiles_to_align = 1:nb_tiles;
    if ~isempty(config.align_tiles) && all(ismember(config.align_tiles,tiles_to_align))
        tiles_to_align = config.align_tiles;
    elseif ~isempty(config.align_tiles) && ~all(ismember(config.align_tiles,tiles_to_align))
        error("Selected subset of tiles to align outside range of all tiles")
    end
    
    % If loading parameters, set image saving to false
    if isequal(config.load_alignment_params,"true")
        config.save_images = "false";
    end

    % Perform channel alignment
    for idx = tiles_to_align
        [y,x] = find(position_mat==idx);            
        fprintf("%s\t Aligning channels by translation to %s for tile %d x %d \n",...
                    datetime('now'),config.markers{1},y,x);
        path_align = path_table(path_table.x==x & path_table.y==y,:);
        aligned_tile = align_by_translation(config,path_align,z_displacement_align);
                
        % Save updated table rows/columns
        if isequal(config.channel_alignment,"update") &&...
                ~isempty(alignment_table{y,x})
            if ~isempty(config.align_slices)
                r_idx = config.align_slices{:};
            else
                r_idx = true(1,height(aligned_tile));
            end
            if ~isempty(config.align_channels)
                c_idx = contains(aligned_tile.Properties.VariableNames,align_markers);
            else
                c_idx = true(1,length(aligned_tile.Properties.VariableNames));
            end
            alignment_table{y,x}(r_idx,c_idx) = aligned_tile(r_idx,c_idx);
            alignment_table{y,x}(r_idx,[1,config.align_channels]) = aligned_tile(r_idx,[1,config.align_channels]);
        else
           alignment_table{y,x} = aligned_tile;
        end
        save(save_path,'alignment_table')
        
        % Save samples
        if isequal(config.save_samples,"true") && ~isempty(config.align_tiles)
            fprintf('%s\t Saving samples \n',datetime('now'));
            save_samples(config,'alignment',path_align)
        end
    end
    
elseif isequal(config.align_method,'elastix')
    fprintf("%s\t Aligning channels using B-splines \n",datetime('now'));

    % Warn user if save_images flag is off. Can't apply params during
    % stitching
    if isequal(config.save_images,"false") && isequal(config.load_alignment_params,"false")
        warning("save_image flag is set to ""false"". Images need to be saved "+...
            "for downstream processing after elastix alignment.")
        pause(5)            
    end

    % Create variable
    alignment_params = cell(nrows,ncols);
    save_path = fullfile(config.output_directory,'variables','alignment_params.mat');
    if ~exist(save_path,'file')
        save(save_path,'alignment_params','-v7.3')
    else
        load(save_path,'alignment_params')
        assert(size(alignment_params,1) == nrows, "Number of row positions do not "+...
            "match loaded alignment parameters")
        assert(size(alignment_params,2) == ncols, "Number of column positions do not "+...
            "match loaded alignment parameters")
    end

    % Define which tiles to align if only doing subset
    tiles_to_align = 1:nb_tiles;
    if ~isempty(config.align_tiles) && all(ismember(config.align_tiles,tiles_to_align))
        tiles_to_align = config.align_tiles;
    elseif ~isempty(config.align_tiles) && ~all(ismember(config.align_tiles,tiles_to_align))
        error("Selected subset of tiles to align outside range of all tiles")
    end

    % Check for intensity thresholds which are required for alignment
    if isempty(config.lowerThresh) || isempty(config.upperThresh) || isempty(config.signalThresh)
        config = check_for_thresholds(config,path_table);
    end

    % If pre-aligning, check for and load calculated translation parameters
    alignment_table = [];
    if isequal(config.pre_align,"true")
        if exist(fullfile(config.var_directory,'alignment_table.mat'),'file') == 2
            load(fullfile(config.var_directory,'alignment_table.mat'),'alignment_table')
            if any(any(cellfun(@(s) isempty(s), alignment_table(position_mat)),'all'))
                idx = find(cellfun(@(s) isempty(s), alignment_table(position_mat)));
                warning("Missing alignment tables at tiles %s.",...
                    string(regexprep(num2str(idx'),' ,','')))
                pause(1)
            end
        else
            fprintf("%s\t Could not locate alignment table for pre-alignment. Generating "+...\
                "new table for pre-alignment.\n",datetime('now'));
            config_pre = config;
            config_pre.align_method = "translation";
            config_pre.align_tiles = tiles_to_align;
            config_pre.save_images = "false";
            config_pre.only_pc = "false";
            config.align_stepsize = 50;
            perform_channel_alignment(config_pre,path_table,equal_res);    
            load(fullfile(config.var_directory,'alignment_table.mat'),'alignment_table')
        end
    end

    % Perform alignment
    for i = tiles_to_align
        [y,x] = find(position_mat==i);
        path_align = path_table(path_table.x==x & path_table.y==y,:);
        if ~isempty(alignment_table)
            config.alignment_table = alignment_table{y,x};
        end
        fprintf("%s\t Aligning channels to %s for tile 0%dx0%d \n",...
            datetime('now'),config.markers{1},y,x);

        alignment_results = elastix_channel_alignment(config,path_align,true);
        %disp(alignment_results(end,:))
        
        % Save with the exception of only aligning slices
        if isequal(config.channel_alignment,"update") && ~isempty(config.align_slices)
            continue
        else
            alignment_params{y,x} = alignment_results;
            save(save_path,'alignment_params','-v7.3')
        end

        % Save samples
        if isequal(config.save_samples,"true") && ~isempty(config.align_tiles)
            fprintf('%s\t Saving samples \n',datetime('now'));
            path_align = path_to_table(config,'aligned',false,false);
            path_align = path_align(path_align.x==x & path_align.y==y,:);

            save_samples(config,'alignment',path_align)
        end
    end
elseif isequal(config.align_method,'merge')
    % Merge images aligned by translation and by elastix into aligned
    % folder. Tiles aligned by elastix assumed to be save in aligned
    % folder. Align all remaining tiles and save into aligned folder
    
    % Load alignment table if it exists
    save_path = fullfile(config.output_directory,'variables','alignment_table.mat');
    if isfile(save_path)
        fprintf("%s\t Loading alignment table that already exists \n",datetime('now'));
        load(save_path,'alignment_table')
        % Check rows and columns
        assert(all(size(alignment_table) == [nrows,ncols]), "Loaded alignment table is not the same shape "+...
            "as the tiles in the input image directory")
    else
        error("No alignment table variable exists")    
    end
    
    % Full tile missing
    config.save_images = "true";
    config.load_alignment_params = "false";
    config.channel_alignment = "true";
    
    % Check images present in aligned folder
    align_f = munge_aligned(config);
    
    % Create grid of tile combinations
    [r,c] = meshgrid(1:nrows,1:ncols);
    d=cat(2,r',c');
    d=reshape(d,[],2);
    %z = unique(path_table(path_table.markers == config.markers(1),:).z);
    
    % Subset markers 
    if isempty(config.align_channels)
        align_markers = config.markers;
    else
        align_markers = config.markers(config.align_channels);
    end   
    align_channels1 = config.align_channels;
    align_markers1 = config.markers;
    
    for i = 1:size(d,1)
        % For each tile, check missing markers/z sections
        path_align = align_f(align_f.x==d(i,2) & align_f.y==d(i,1),:);
        
        if ~isempty(path_align) &&...
                all(ismember(align_markers,path_align.markers))
            % All markers for this tile already exist
            continue
        elseif ~all(ismember(align_markers,path_align.markers))
            align_channels = find(~ismember(align_markers1,path_align.markers));
            config.align_channels = align_channels(ismember(align_channels,align_channels1));
            align_markers = align_markers1(config.align_channels);
        else
            config.align_channels = align_channels1;
            align_markers = align_markers1;
        end
               
        % Align this tile
        fprintf("%s\t Merging tile %d x %d for markers %s\t \n",datetime('now'),...
            d(i,1),d(i,2),strjoin(align_markers))   
        
        path_sub = path_table(path_table.x==d(i,2) & path_table.y==d(i,1),:);
        align_by_translation(config,path_sub);
    end
else
    error("Unrecognized selection for align_method. "+...
        "Please select ""translation"", ""elastix"", ""merge"".")
end

% Save samples on full
if isequal(config.save_samples,"true")
    fprintf('%s\t Saving samples \n',datetime('now'));
    save_samples(config,'alignment')
end

fprintf("%s\t Alignment for sample %s completed! \n",datetime('now'),config.sample_id);

% Change image directory to aligned directory so that subsequent
% steps load these images. Otherwise point config to alignment params
config.load_alignment_params = "false";
if isequal(config.save_images,"true")
    config.img_directory = fullfile(config.output_directory,"aligned");
    path_table = path_to_table(config,"aligned",false);
    
    % Check number of loaded tiles
    ncols = length(unique(path_table.x));
    nrows = length(unique(path_table.y));
    if nb_tiles ~= ncols*nrows
        warning("Number of aligned tiles and raw image tiles are not equal")
        pause(5)
    end
    
    % Update tile intensity adjustments using newly aligned images.
    % Also, set light sheet width adjustments + flatfield adjustments
    % to false as these were applied during the alignment step
    config.adjust_tile_shading = repmat("false",1,length(config.markers));
    config.adj_params.adjust_tile_shading = config.adjust_tile_shading;
    
    % In case applying tile adjustments, re-calculate thresholds
    if isequal(config.adjust_intensity,"true") && isequal(config.adjust_tile_position,"true")
        for k = 1:length(config.markers)            
            fprintf("%s\t Updating intensity measurements for marker %s using "+...
                "newly aligned images \n",datetime('now'),config.markers(k));
            stack = path_table(path_table.markers == config.markers(k),:);
            [~,~,~,t_adj] = measure_images(config, stack, k);
            config.adj_params.t_adj{k} = t_adj;
        end
    end
    
elseif isequal(config.align_method,'translation') && isequal(config.save_images,"false")
    config.load_alignment_params = "true";
end

end