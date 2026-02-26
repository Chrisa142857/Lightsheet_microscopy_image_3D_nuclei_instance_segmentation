function tblSATH = measure_cortical_sa_th(input,res,use_l1_border,flatview,save_table)
%--------------------------------------------------------------------------
% Calculate the volume, surface area, and thickness of each major structure
% in the cortex (according to Harris et al, Nature 2019. Input is a
% registered mask of cortical annotations (I_mask). I_mask should contain
% layer-specific annotations. Output measurements are in mm.
%--------------------------------------------------------------------------

% Default resolution 25um/voxel
if nargin<2
    res = 25;
end

% Whether to use L1 surface for SA calculations. Otherwise, entire cortical
% suface area (including inner L6 border)
if nargin<3
    use_l1_border = 'true';
end

% Use 17 or 43 or both cortical structures
if nargin<4
    flatview = 'both';
end

if nargin<5
    save_table = false;
end
home_path = fileparts(which('NM_config'));

% If string, find all mask files in directory
if ischar(input) || isstring(input)
    if isfolder(input)
        files = dir(input);
        files = string(fullfile(files.folder,{files.name}));
    else
        files = input;
    end
    files = files(arrayfun(@(s) endsWith(s,".mat"),files));
    n_samples = length(files);
    if isempty(n_samples)
        error("First input should be a single 3D image of ARA annotations or " +...
            " a config structure from NM_evaluate to measure multiple samples")
    end
    
elseif isstruct(input)
    if ~isfield(input,'main_stage') || ~isequal(input.main_stage,'evaluate')
        error("If providing structure, call NM_config('evaluate',group) to generate "+...
            "correct configuration structure for running multiple samples")
    end
    files = fullfile(input.centroids_directory,'variables',strcat(input.samples','_mask.mat'));
    n_samples = length(files);
    
else
    n_samples = 1;
    sample_name = 'SAMPLE';
end

% Read cortical structure indexes
all_struct = readtable(fullfile(home_path,'annotations','structure_template.csv'));
if isequal(flatview,'seventeen')
    ctx_harris = readtable(fullfile(home_path,'annotations','custom_annotations','cortex_17regions.xls'));
    adj = 1;
elseif isequal(flatview,'harris')
   ctx_harris = readtable(fullfile(home_path,'annotations','custom_annotations','harris_cortical_groupings.xls'));
   adj = 1;
elseif isequal(flatview,'both')
    ctx1 = readtable(fullfile(home_path,'annotations','custom_annotations','cortex_17regions.xls'));
    ctx2 = readtable(fullfile(home_path,'annotations','custom_annotations','harris_cortical_groupings.xls'));
    ctx_harris = [ctx1;ctx2];
    ctx_harris = unique(ctx_harris,'rows');
    adj = 2;
end

% Get cortex ids from structure path 
all_ids = all_struct.structure_id_path;
a = cellfun(@(s) strsplit(s,'/'),all_ids,'UniformOutput',false);
a = cellfun(@(s) str2double(s), a, 'UniformOutput',false);
a = cellfun(@(s) s(~isnan(s)), a, 'UniformOutput',false);

% Get ids for major cortical structures from harris et al. 2019
harris_ids = ctx_harris.id;
n_structures = length(harris_ids);

for i = 1:n_samples
    if ischar(input) || isstring(input) || isstruct(input)
        [~,sample_name] = fileparts(files(i));
        sample_name = extractBefore(sample_name,"_");
        fprintf("Reading sample %s\n",sample_name)
        load(files(i),'I_mask');
        if endsWith(files(i),"_results.mat")
            res_struct = true;
        else
            res_struct = false;
        end
    else
        I_mask = input;
        res_struct = false;
    end

    % Unique indexes in volume
    all_indexes = unique(I_mask);

    % Calculate border image
    BW = I_mask>0;
    BWe = imerode(BW,ones(3,3,3));
    I_border = I_mask.*(BW~=BWe);
    
    % Do outer?
    if isequal(use_l1_border,'true')
        % Read l1 indexes
        fprintf("Using pial surface for SA calculation\n")
        l1_tbl = readtable(fullfile(home_path,'annotations','default_annotations','cortical_layers','layer1.csv'));
        BW2 = I_mask;
        BW2(~ismember(BW2,l1_tbl.index)) = 0;
        BW2 = BW2>0;
    else
        fprintf("Using full cortex surface for SA calculation %s\n",files(i).name)
    end

    % Calculate total surface area
    if isequal(use_l1_border,'true')
        total_sa = regionprops3(BW2,'SurfaceArea');
        total_sa = max(total_sa.SurfaceArea);
        BW_L1 = BW2 & I_border;
        f = sum(BW_L1(:))/sum(BW2(:));
        total_sa = total_sa*f;
    else
        total_sa = regionprops3(BW,'SurfaceArea');
        total_sa = max(total_sa.SurfaceArea);
    end
    
    total_border = sum(I_border(:)>0);

    % Calculate centroid positions for all structures
    all_cen = regionprops3(I_mask,'Centroid');
    all_cen = all_cen.Centroid;

    % For each structure, calculate surface area, thickness, and volume
    vol = zeros(1,n_structures);
    th = vol;
    sa = vol;
    for j = 1:n_structures
       %fprintf('Working on structure %d out of %d\n',j,n_structures)

       % Get ids for this structure
       b = cellfun(@(s) any(ismember(s,harris_ids(j))), a);
       struct_idx = all_struct(b,:).index;
       struct_idx = struct_idx(ismember(struct_idx,all_indexes));

       % Subset this structure
       I_struct = I_mask.*ismember(I_mask,struct_idx);

       % Get volume
       vol(j) = sum(I_struct(:)>0);

       % Get surface area
       % Calculate the fraction of voxels from this structure that contribute 
       % to the total border image. Then multiply this fraction by the total 
       % surface area to get surface area for just this structure.
       if isequal(use_l1_border,'true')
           % Fraction L1 along the border for this structure
           f = sum(ismember(I_border(BW_L1),struct_idx))/sum(BW_L1(:));  
       else
           f = sum(ismember(I_border(:),struct_idx))/total_border;
       end
       sa(j) = total_sa*f;

       % Get thickness
       % Find the voxels around the border closest to the centroids for layers
       % 1 and 6. Find the centroid of the entire structure. Calculate the
       % spline curve length that runs through all 3 points.
       % Note layer indexes are by default ordered from layer 1 to layer 6
       l1_idx = min(struct_idx);
       c1 = all_cen(l1_idx,:);
       % Find nearest l1 voxel
       [x,y,z] = ind2sub(size(I_border),find(I_border == l1_idx));
       idx = dsearchn([y,x,z],c1);
       c(1,:) = [y(idx), x(idx), z(idx)];
       % Find centroid of whole structure
       s_cen = regionprops3(I_struct>0,'Volume','Centroid');
       idx = s_cen.Volume == max(s_cen.Volume);
       c(2,:) = s_cen.Centroid(idx,:);
       % Find nearest l6 voxel
       l6_idx = max(struct_idx);
       c6 = all_cen(l6_idx,:);
       [x,y,z] = ind2sub(size(I_border),find(I_border == l6_idx));
       idx = dsearchn([y,x,z],c6);
       %c(3,:) = [y(idx), x(idx), z(idx)];
       c(2,:) = [y(idx), x(idx), z(idx)];

       % Calculate arc length
       th(j) = arclength(c(:,1),c(:,2),c(:,3),'spline');
       th(j) = pdist(c,'euclidean');
    end

    % Apply scaling
    th = th*(res/1E3);
    sa = sa*(res/1E3)^2;
    vol = vol*(res/1E3)^3;
    
    % Save to structure
    if res_struct
        cortex.index = ctx_harris.index;
        cortex.volume = vol';
        cortex.sa = sa';
        cortex.th = th';
        save(files(i),'-append','cortex')
    end
    
    if nargout<1
        % Save results to table
        tbl = table();
        tbl.index = ctx_harris.index;
        tbl.Sample = repmat({sample_name},1,n_structures)';
        tbl.Acronym = ctx_harris.acronym;
        tbl.Volume = vol';
        tbl.SA = sa';
        tbl.TH = th';
        if i ==1
            tblSATH = tbl;
        else
            tblSATH = vertcat(tblSATH,tbl);
        end
    end
    
    fprintf("Volume, SA, TH: %f %f %f\n" ,sum(vol)/adj,sum(sa)/adj,mean(th))
end

if save_table
    writetable(tblSATH,'ctxVolSATH_measurements.csv')
end

end


function [vol, sa, th] = use_isosurface(I_mask, l_idx)
% Calculate surface after fitting a 3D triangular mesh

% Get binary mask of full cortex
BW = I_mask>0;

% Volume is just sum of pixels
vol = sum(BW(:));

% Erode mask to expose surface voxels
BWe = imerode(BW,ones(3,3,3));

% Get border pixels 
BWp = BW ~= BWe;
%[p_x,p_y,p_z] = ind2sub(size(BWp),find(BWp == 1));

% Fit isosurface and reduce vertices to make things faster
s = isosurface(BW,0);
s = reducepatch(s,0.1);

% Smooth isosurface
s2 = smoothpatch(s,1,1,1);
faces = s2.faces;
verts = s2.vertices;

% Get annotation indexes at positions
verts1 = round(verts);
idx = sub2ind(size(BW),verts1(:,2),verts1(:,1),verts1(:,3));
I_mask_exp = imdilate(I_mask,ones(3,3,3));
a_idx = I_mask_exp(idx);

% Subset the layer 1 faces to get outer surface area
l1_idx = find(a_idx == l_idx(1));
l1_verts = verts(l1_idx,:);
new_idx = 1:size(l1_verts,1);
l1_faces = faces(all(ismember(faces,l1_idx),2),:);
l1_faces = changem(l1_faces,new_idx,l1_idx);




end

function sa = calculate_sa(verts, faces)

% Calculate area of each triangle, and sum them to get surface area
a = verts(faces(:, 2), :) - verts(faces(:, 1), :);
b = verts(faces(:, 3), :) - verts(faces(:, 1), :);
c = cross(a, b, 2);
sa = 1/2 * sum(sqrt(sum(c.^2, 2)));

end

