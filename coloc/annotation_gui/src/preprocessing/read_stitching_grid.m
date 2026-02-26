function A = read_stitching_grid(img_grid,stitch_channels,markers,adj_params,alignment_table)
%--------------------------------------------------------------------------
% Read cell grid of images and apply adjustments prior to stitching. Return
% as single.
%--------------------------------------------------------------------------

[nrows, ncols,nchannels] = size(img_grid);
A = cell(nrows,ncols,nchannels);
tempI = read_img(img_grid{1});
[img_height,img_width] = size(tempI);
ref_fixed = imref2d([img_height img_width]);

c_idx = zeros(1,nchannels);
c_idx(stitch_channels) = 1;

% Read grid
for i = 1:nrows
    for j = 1:ncols
        for k = 1:nchannels                        
            % Applying alignment transforms
            if ~isempty(alignment_table) && k==1 
                % Get row index of image
                [path,r1,ext] = fileparts(img_grid{i,j,1});
                r_idx = find(cellfun(@(s) endsWith(s,strcat(r1,ext)),alignment_table{i,j}{:,1}));
                if c_idx(k)==1
                    % Read reference images
                    A{i,j,k} = read_img(img_grid{i,j,k});
                else
                    continue
                end
            elseif ~isempty(alignment_table) && k>1 && c_idx(k)==1 
                % Read secondary channel image from alignment table
                if string(alignment_table{i,j}{r_idx,k}) == ""
                    A{i,j,k} = zeros(img_height,img_width);
                    continue
                else
                    [~,b,c] = fileparts(string(alignment_table{i,j}{r_idx,k}));
                    align_img = fullfile(path,strcat(b,c));
                    if b ~= ""
                        A{i,j,k} = read_img(align_img);
                    else
                        A{i,j,k} = zeros(img_height,img_width);
                        continue
                    end
                end
            elseif isempty(alignment_table) && c_idx(k)==1 
                % Read image
                A{i,j,k} = imread(img_grid{i,j,k});
            else
                continue
            end
                    
            % Crop or pad images
            if ~isequal(size(A{i,j,k}),[img_height,img_width])
                A{i,j,k} = crop_to_ref(tempI,A{i,j,k});
            end
            
            % Apply intensity adjustments
            if ~isempty(adj_params)
               % Crop laser width adjustment if necessary
                if length(adj_params{k}.y_adj) ~= length(adj_params{1}.y_adj)
                    adj_params{k}.y_adj = crop_to_ref(adj_params{k}.y_adj,...
                        adj_params{1}.y_adj);
                    A{i,j,k} = crop_to_ref(tempI,A{i,j,k});
                end
                % Apply adjustments
                A{i,j,k} = apply_intensity_adjustment(A{i,j,k},adj_params{k},...
                    'r',i,'c',j);                
            end
            
            if ~isempty(alignment_table) && k>1 && c_idx(k)==1                
                % Get x,y translations and shift
                x = alignment_table{i,j}{r_idx,"X_Shift_"+markers(k)};
                y = alignment_table{i,j}{r_idx,"Y_Shift_"+markers(k)};
                
                tform = affine2d([1 0 0; 0 1 0; x y 1]);
                A{i,j,k} = imwarp(A{i,j,k}, tform,'OutputView',ref_fixed,'FillValues',0); 
            end
            
        end
    end
end
            
%Convert images to single

A = A(:,:,logical(c_idx));
A = cellfun(@(s) single(s),A,'UniformOutput',false);
            
end