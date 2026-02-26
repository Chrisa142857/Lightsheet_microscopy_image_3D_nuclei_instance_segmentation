function vox = create_stats_volume(df_stats, voxel_imgs, markers, plot_type)
% Create a new annotation volume where each structure index is colored with
% a new field from the summary stats table
home_path = fileparts(which('NM_config.m'));
av = load(fullfile(home_path,'data','annotation_data','ccfv3.mat'));
downsample_factor = 0.1;
nmarkers = length(markers);

% Permute the annotation volume based on desired brain orientation
if isequal(plot_type,"sagittal")
    av.annotationVolume = permute(av.annotationVolume,[1,3,2]);
elseif isequal(plot_type, "axial")
    av.annotationVolume = permute(av.annotationVolume,[3,2,1]);
end

% Add annotation info for all structures
temp_tbl = readtable(fullfile(home_path,'annotations','structure_template.csv'));
vox.info = table2struct(temp_tbl(:,1:9));

% Subset slices to reduce data sizes and make things smoother
[nrows,ncols,nslices] = size(av.annotationVolume);
nslices = round(nslices*downsample_factor);
av.annotationVolume = imresize3(av.annotationVolume,[nrows,ncols,nslices],'Method','nearest');

% Begin building visualization structure
% Add structure info + initial slice positions
vox.z = round(size(av.annotationVolume,3)/2);
vox.bregma = (540/1320)*downsample_factor;
vox.marker_names = markers;

% Add colors with indexes for major structure divisions in Allen reference
c = load(fullfile(home_path,'src','utils','allen_ccf_colormap_2017.mat'));
vox.av_colors = c.cmap;
vox.av_cmap_labels = ["Isocortex","OLF","HPF","CTXsp","STR","PAL","TH","HY","MB","HB","CB","FT"];
vox.av_cmap_pos = [5,379,454,555,573,608,641,715,806,882,1014,1101];

% Create voxel images
for i = 1:length(markers)
    img = voxel_imgs([voxel_imgs.marker] == markers(i)).data;
    if isequal(plot_type,"coronal")
        img = permute_orientation(img,"ail","sra");
    elseif isequal(plot_type,"sagittal")
        img = permute_orientation(img,"ail","sal");
    elseif isequal(plot_type,"axial")
        img = permute_orientation(img,"ail","als");
    end
    for j = 1:size(img,4)
        if j == 1
            a = squeeze(img(:,:,:,j));
            a(a>1) = 1;
            vox.voxel(i).img{j} = a*100;
        else
            a = squeeze(img(:,:,:,j));
            a(a>5) = 5;
            vox.voxel(i).img{j} = a;
        end
    end
end

% Bin to correct structures
f = fieldnames(df_stats);
for i = 1:length(f)
    if isequal(f{i},'Cortex')
        continue
    end
    s = trim_to_deepest_structure(df_stats.(f{i}));
    if i == 1
        df_stats_new.info = s.info;
        df_stats_new.category = string(f{i});
        df_stats_new.stats = s.stats;
    else
        df_stats_new.category = cat(2,df_stats_new.category,repmat(string(f{i}),1,length(s.stats)));
        df_stats_new.stats = cat(2,df_stats_new.stats,s.stats);
    end
end

[aV,aI] = bin_annotation_structures(av.annotationVolume,df_stats_new.info,'true');
binnedMask = ismember(aV,aI);
% Potentially edit
vox.av = aV+1;
vox.indexes = aI;
vox.binMask = binnedMask;

% Precompute annotation indexes for each slice
% Add boundary image for each slice
% Add AP position of Bregma
vox.boundaries = zeros(size(av.annotationVolume),'logical');
vox.av_stat_idx = cell(1,nslices);
for i = 1:nslices
    vox.av_idx(i) = {unique(vox.av(:,:,i))};
    vox.av_stat_idx{i} = find(ismember(aI+1,vox.av_idx{i}));
    vox.boundaries(:,:,i) = boundarymask(aV(:,:,i),8);
    if isequal(plot_type,"coronal")
        vox.ap(i) = ((540*downsample_factor) - i)/(100*downsample_factor);
    end
end

vox.bin_idx = cellfun(@(s) ismember(s-1,av.annotationIndexes),vox.av_idx,...
    'UniformOutput',false);

% Now for each marker, add counts and densities
for i = 1:length(df_stats_new.stats)  
    if i == 1
        idx = 1;
        marker = "Volume";
    else
        idx = find(df_stats_new.stats(i).marker == markers);
        marker = replace(df_stats_new.stats(i).marker,{'/','.'},'');
    end
    vox = add_stats_to_vs(vox,df_stats_new.stats(i).stats,marker,df_stats_new.category(i),idx);
end

% Update indexes to match volume
vox.indexes = vox.indexes+1;

end


function vox = add_stats_to_vs(vox,values,marker,colname,idx)

% Add calculated statistics for each marker
vox.markers(idx).(colname).title = marker + " " + colname;
vox.markers(idx).(colname).name = colname;

vox.markers(idx).(colname).values(:,1:5) = values(:,1:5);
vox.markers(idx).(colname).values(:,6:7) = -log10(values(:,6:7));
vox.markers(idx).(colname).values(:,8) = values(:,8);

vox.markers(idx).(colname).cmin = min(values);
vox.markers(idx).(colname).cmin(6:8) = 0;
vox.markers(idx).(colname).cmin(6:8) = 1.301;
vox.markers(idx).(colname).cmax = max(values);
vox.markers(idx).(colname).cmax(6:7) = 5;
vox.markers(idx).(colname).cmax(8) = 4;  

pchange_max = max(abs(vox.markers(idx).(colname).cmin(5)),vox.markers(idx).(colname).cmax(5))*1.1;
%vox.markers(idx).(colname).cmin(5) = -pchange_max;
%vox.markers(idx).(colname).cmax(5) = pchange_max;

vox.markers(idx).(colname).cmin(5) = -100;
vox.markers(idx).(colname).cmax(5) = 100;

% Set colors
vox.markers(idx).colors(1:4) = {brewermap(201,'YlOrBr')};    
vox.markers(idx).colors(5) = {brewermap(201,'*RdBu')};
vox.markers(idx).colors(6:7) = {brewermap(401,'*plasma')};    
vox.markers(idx).colors(8) = {brewermap(5,'Purples')};
    
end
