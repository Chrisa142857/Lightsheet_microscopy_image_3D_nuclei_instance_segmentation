function [h, max_val, min_val] = plot_flat_cortex(input,p_values,stat,flatview,limits,thresh,rescale)
%--------------------------------------------------------------------------
% Visualize flattened cortex at 10um resolution. 
%--------------------------------------------------------------------------
% Usage:
% visualize_flat_cortex(input, key, flatview)
%
%--------------------------------------------------------------------------
% Inputs:
% input:    1. Flatmap structure containg voxelized cortex data for heatmap
%              representation. 
%           2. Table containing fold change and p values by cortical 
%              annotation index.
%           3. 2D image at the same size as 10um flatmap (1360x1360)
%
% key: (1x3 integer) First index indicates cell class index. Second index
% indicates statistic (1=fold change, 2=p-value, 3=p.adj). Third index
% indicates thresholding signficant regions (1=true, 0=false).
% 
% flatview: ('harris' or 'seventeen') Bin cortical structures. 
%
%--------------------------------------------------------------------------

%rescale_flag = true;
n_colors = 201;
filter_size = 51;
stat = 1;

if nargin<3
    stat = 1;
end

if nargin<4
    flatview = 'harris';
end

if nargin<5
    limits = [-100,Inf];
end

if nargin<6
    thresh = 'none';
end

if nargin<7
    rescale = false;
end

if stat == 1
    adj = 100;
    [~,fname] = fileparts(input);
    fname = strsplit(fname,'_');
    t_disp = sprintf("%s %s Percent Change",fname{1},fname{2});
elseif stat == 2
    limits = [0,5];
    adj = 1;
    t_disp = "p Value";
elseif stat == 3
    limits = [1.301,5];
    adj = 1;
    t_disp = "Adjusted p Value";
else
    error("Stat index should be 1, 2, or 3")
end

switch thresh 
    case 'none'
        thresh = [];
    case 'p'
        thresh = 2;
    case 'p_adj'
        thresh = 3;
end

%figure;imagesc(results2)

% Load flatview cortex
fc = load(fullfile(fileparts(which('NM_config')),'data','annotation_data','flatviewCortex.mat'),flatview);
if isequal(flatview,'harris')
    fc = fc.harris;
    ftable = readtable(fullfile(fileparts(which('NM_config')),'annotations','custom_annotations','harris_cortical_groupings.xls'));
else
    fc = fc.seventeen;
    ftable = readtable(fullfile(fileparts(which('NM_config')),'annotations','custom_annotations','cortex_17regions.xls'));
end

% Read cortical groupings
indexes = ftable.index;
if isstruct(input)
    img = squeeze(input(key(1)).data(:,:,key(2)));
    
    if key(3) == 1
        sig_bw = squeeze(input(key(1)).data(:,:,3));
        sig_bw = imgaussfilt(sig_bw,15);
        sig_bw = -log10(sig_bw)<1.301;
    else
        sig_bw = [];
    end
    
    if key(2) == 1
        img = img*100-100;
    else
        img = -log10(img);
    end
    img = imgaussfilt(img,15);
    img(sig_bw) = 0;
    t_disp = input(key(1)).marker;
    
elseif istable(input)
    input = input(ismember(input.index,indexes),:);
    v = input.(beta_col);
    if ~isempty(p_col)
        p = input.(p_col);
        %v(p>0.05) = 1;
        sig = p<0.05;
    end
    i_index = input.index;
    [~, i] = sort(i_index);
    v = v(i);
    if ~isempty(p_col)
        sig = sig(i);
    end
    
    % Round vector to nearest hundreth
    v = round(v,2);
    v(v>2) = 2;
    img = zeros(size(fc.flatCtx));
    for i = 1:length(indexes)
        img(fc.flatCtx==indexes(i)) = v(i);
    end
elseif isnumeric(input) || ischar(input) || isstring(input)
    % Read image
    if ischar(input) || isstring(input)
        img0 = niftiread(input);
        img = squeeze(img0(:,:,stat));
    else
        img = input;
    end
    img(isnan(img)) = 0;
    img = round(img*adj);

    % Threshold significant regions
    if ~isempty(thresh)
        sig_bw = squeeze(img0(:,:,thresh));
        %sig_bw = imgaussfilt(sig_bw,3);
        %sig_bw = sig_bw >1.300;
        img = sig_bw;
    else
        sig_bw = true(size(img));
    end
    
    % Set limits
    max_val = max(img(:));
    min_val = min(img(:));
    if limits(2) == Inf
        limits(2) = max_val;
    end
    
    % Smooth image
    img = imgaussfilt(img,filter_size,'FilterSize',21);
end

% Calculate p_values
if nargin>2 && ~isempty(p_values)
    [~,~,~,adj_p] = fdr_bh(p_values(:,2));
    %p_values = p_values(ismember(p_values(:,1),indexes),2);
    adj_p = adj_p(ismember(p_values(:,1),indexes));
    sig = adj_p<0.05;
end

% Get mask and boundaries
boundaries = fc.boundaries;
bw = fc.flatCtx==0 & ~boundaries;
if ishandle(1)
    close(gcf)
end
ax1 = axes;
imagesc(img);

% Load colors
colors = brewermap(n_colors,'*RdBu');
s = round(linspace(-100,100,n_colors));
mid = (n_colors-1)/2 + 1;
if rescale
    i = ismember(s,limits(1):limits(2));
    
    i1 = sum(i(1:mid));
    i2 = sum(i(mid:end));
    
    if i1 > 1
        colors1 = interp1(1:mid,colors(1:mid,:),linspace(1,mid,i1));
    else
        colors1 = [];
    end
    
    if i2 > 1
        colors2 = interp1(1:mid,colors(mid:n_colors,:),linspace(1,mid,i1));
    else
        colors2 = [];
    end
    colors = cat(1,colors1,colors2);
else
    colors = colors(ismember(s,limits(1):limits(2)),:);
end

colors = brewermap(n_colors,'*plasma');
limits = [1.3,1.5];

colormap(colors)
caxis([limits(1) limits(2)])
set(ax1,'XTick',[], 'YTick', [])
daspect([1,1,1])

view(2)
ax0 = axes;
a0 = image(sig_bw);
a0.AlphaData = ~sig_bw;
ax0.Colormap = [[1,1,1];[0,0,0]];
ax2 = axes;
a2 = image(bw);
a2.AlphaData = bw;
ax2.Colormap = [[0,0,0];[1,1,1]];
ax3 = axes;
a3 = image(fc.boundaries);
a3.AlphaData = fc.boundaries;
ax3.Colormap = [[1,1,1];[0,0,0]];
linkaxes([ax1,ax0,ax2,ax3])

ax1.Visible = 'off';
ax1.XTick = [];
ax1.YTick = [];
ax0.Visible = 'off';
ax0.XTick = [];
ax0.YTick = [];
ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];
ax3.Visible = 'off';
ax3.XTick = [];
ax3.YTick = [];

P = get(ax1,'Position');
XLIM = get(ax1,'XLim');
YLIM = get(ax1,'YLim');
PA = get(ax1,'PlotBoxAspectRatio');

set(ax0,'Position',P,'XLim',XLIM,'YLim',YLIM,'PlotBoxAspectRatio',PA)
set(ax2,'Position',P,'XLim',XLIM,'YLim',YLIM,'PlotBoxAspectRatio',PA)
set(ax3,'Position',P,'XLim',XLIM,'YLim',YLIM,'PlotBoxAspectRatio',PA)

% Add structure names
x = fc.x;
y = fc.y;
for i = 1:length(indexes)
    if isstruct(input) || isempty(p_values) || ~sig(i)
        text(x(i),y(i),fc.acronym{i},'FontName','Arial','FontSize',12)
    else
        text(x(i),y(i),strcat(fc.acronym{i},'*'),...
            'FontName','Arial','FontSize',12,'FontWeight','bold')
    end
end
text(0,-30,t_disp,'FontName','Arial','FontSize',12,'FontWeight','bold')

set(gcf,'CurrentAxes',ax1)
h = colorbar;
h.Limits = [limits(1),limits(2)];
h.Ticks = limits(1):diff(limits)/4:limits(2);
h.Position(1) = 0.85;

h = gcf;

end
