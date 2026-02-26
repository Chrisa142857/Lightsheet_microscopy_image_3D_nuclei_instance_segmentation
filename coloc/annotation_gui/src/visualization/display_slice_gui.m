function [p1,p2] = display_slice_gui(vs,key)
% User defined keys
% z: z_position
% marker: which marker
% category: Counts, volume, Density
% stat: which stat
% key: [z_positition, marker, stat];

if key{1}(3) == 1 && key{1}(2) ~= 0
    key{1}(2) = 1;
end
if key{2}(3) == 1 && key{1}(2) ~= 0
    key{2}(2) = 1;
end

% User data structure
ud.z = key{1}(1);
ud.marker = [key{1}(2),key{2}(2)];
ud.category = [key{1}(3),key{2}(3)];
ud.stat = [key{1}(4),key{2}(4)];
ud.pos = 1;
ud.alpha = 'false';
ud.alpha_val = 0.9;
ud.init = 'true';
ud.imgL = zeros(size(vs.av(:,:,1)));
ud.imgR = zeros(size(vs.av(:,:,1)));

f = figure;
f.Position = [5,600,855,600]; %AR = 1.425
f.Position = [5,600,855*1.17,600*1.17]; %AR = 1.425

%f.Position = [5,600,741,520];
%f.Position = [5,600,1710,1200];

[p1,cmin,cmax,adj] = display_plot(ud,vs,f);
ud.cmin(1) = cmin; ud.cmax(1) = cmax; ud.cmin(1) = cmin; ud.adj(1) = adj;
p1 = sanitize_plot(p1,1);

ud.pos = 2;
[p2,cmin,cmax,adj] = display_plot(ud,vs,f);
ud.cmin(2) = cmin; ud.cmax(2) = cmax; ud.adj(2) = adj;
p2 = sanitize_plot(p2,2);

linkaxes([p1 p2],'xy')

% Set colorbar
ud.pos = 1;
p1 = add_colorbar(p1,vs,ud);
ud.pos = 2;
p2 = add_colorbar(p2,vs,ud);

% Add structure info top left
ud.stext = annotation('textbox', [0 0.95 0.4 0.05], ...
    'EdgeColor', 'none', 'Color', 'k', 'FontSize',12, 'FontWeight','bold');

% Add position info top right
ud.ptext = annotation('textbox', [0.6 0.95 0.4 0.05], ...
    'EdgeColor', 'none', 'Color', 'k','FontSize',12,'HorizontalAlignment','right');

% Add p1 stats to the top left 
ud.p1text = annotation('textbox', [0.0 0.9 0.4 0.05], ...
    'EdgeColor', 'none', 'Color', 'k','FontSize',12,'HorizontalAlignment','left');

% Add p2 stats to the top right
ud.p2text = annotation('textbox', [0.6 0.9 0.4 0.05], ...
    'EdgeColor', 'none', 'Color', 'k','FontSize',12,'HorizontalAlignment','right');

set(ud.ptext, 'String', sprintf('Slice:%d/%d \t %.2f AP', ud.z,size(vs.av,3),vs.ap(ud.z)));

ud.init = 'false';
set(f, 'UserData', ud);
set(f, 'WindowButtonDownFcn',@(f,k)fh_wbmfcn(f, vs)) % Set click detector.
f.PaperOrientation = 'landscape';

end



function [p,cmin,cmax,adj] = display_plot(ud,vs,f)

if isequal(ud.init,'true')
    p = subplot(1,2,ud.pos);
else
    child = get(f,'Children');
    if ud.pos == 1
        p = child(4);
    else
        p = child(2);
    end
end

[slice,colors,title] = retrieve_slice(ud,vs);
if isequal(ud.alpha,'true') && ud.marker(ud.pos) == 0
    a = ~ismember(single(vs.av(:,:,ud.z)),single(vs.indexes));
    slice(a) = 1;
    colors = colors(vs.indexes(vs.av_stat_idx{ud.z}),:);
    [~,~,z] = unique(slice);
    x = 1:length(colors)+1;
    slice(:) = x(z);
end
h1 = imagesc(single(slice));

% Set caxis mappings
if ud.marker(ud.pos) == 0
    cmin = 0; cmax = length(colors); adj = 1;
    h1.CDataMapping = 'direct';
    colormap(p,[[1,1,1];colors;[0.2,0.2,0.2]])
    cmax_adj = cmax+2;
    caxis([cmin cmax_adj])
        
else
    [cmin,cmax] = retrieve_data(vs,ud);
    h1.CDataMapping = 'scaled';  
    if ismember(ud.stat(ud.pos),1:8)
        % Mean, St.Dev. Counts, Volumes, Density....
        colormap(p,[[1,1,1];colors;[0.2,0.2,0.2]])
    end
    
    %if ud.pos == 1
    %   cmin = -100*1.02;cmax=100*1.02; 
    %else
    %    cmin = 1*1.02; cmax = 5*1.02;
    %end
    
    adj = (cmax-cmin)/205;
    h1.CData(~vs.binMask(:,:,ud.z)) = cmin-adj;
    cmax_adj = cmax+(adj*2.01);
    caxis([cmin cmax_adj])
end

% Add title
p.Title.String = title;
p.Title.FontSize = 14;

% Overlay boundaries
hold on

% Add alpha to annotation volume
a = ~ismember(single(vs.av(:,:,ud.z)),single(vs.indexes));
h3 = image(vs.binMask(:,:,ud.z));
h1.CDataMapping = 'scaled';
if ud.marker(ud.pos) == 0
    set(h3,'AlphaData',a*ud.alpha_val)
else
    set(h3,'AlphaData',a)
end
%h3.Visible = 'off';

boundaries = vs.boundaries(:,:,ud.z)*cmax_adj;
if ud.marker(ud.pos) == 0
    h2 = image(boundaries);
else
    h2 = imagesc(boundaries);
end
set(h2,'AlphaData',vs.boundaries(:,:,ud.z))
hold off

end

function [slice,colors,title] = retrieve_slice(ud,vs)

% Get which axis
a = ud.pos;

% Displaying annotation volume
if ud.marker(a) == 0
    slice = vs.av(:,:,ud.z);
    colors = vs.av_colors;
    title = "Annotation Volume";
    return
elseif ud.marker(a) > length(vs.marker_names)
    ud.category(a) = 4;
end
    
% Displaying some statistics
slice = double(vs.av(:,:,ud.z));
idxs = vs.av_idx{ud.z};

if isequal(ud.category(a),1)
    % Do volume
    stats = vs.markers(ud.marker(a)).Volumes.values(:,ud.stat(a));
    title = vs.markers(ud.marker(a)).Volumes.title(ud.stat(a));

elseif isequal(ud.category(a),2)
    % Do Counts
    stats = vs.markers(ud.marker(a)).Counts.values(:,ud.stat(a));
    title = vs.markers(ud.marker(a)).Counts.title;

elseif isequal(ud.category(a),3)
    %  Do Density
    stats = vs.markers(ud.marker(a)).Density.values(:,ud.stat(a));
    title = vs.markers(ud.marker(a)).Density.title;

elseif isequal(ud.category(a),4)
    %  Do Voxel
    slice = vs.voxel(ud.marker(a)).img{ud.stat(a)-4}(:,:,ud.z);
    slice = flip(slice,2);
    slice = imresize(slice,10,'bilinear');
    slice = imgaussfilt(slice,10,'FilterSize',15);
    if ud.stat(ud.pos) == 7
        slice(slice<1) = 1;
    end
    title = vs.markers(ud.marker(a)).Counts.title;
    colors = vs.markers(ud.marker(a)).colors{ud.stat(a)};
    return
    %stats = vs.markers(ud.marker(a)).Custom.values(:,ud.stat(a));
    %title = vs.markers(ud.marker(a)).Custom.title(ud.stat(a));

else
    error("Invalid stats category selected")
end

for i = 1:length(vs.indexes)
    if ismember(vs.indexes(i),vs.av_idx{ud.z})
        slice(slice == vs.indexes(i)) = stats(i);
    end
end
colors = vs.markers(ud.marker(a)).colors{ud.stat(a)};

end

function [cmin,cmax,title,name,values] = retrieve_data(vs,ud)

% Get axis
a = ud.pos;

if ud.marker(a) > length(vs.marker_names)
    ud.category(a) = 4;
end

if isequal(ud.category(a),1)
    % Do volume
    cmin = vs.markers(ud.marker(a)).Volumes.cmin(:,ud.stat(a));
    cmax = vs.markers(ud.marker(a)).Volumes.cmax(:,ud.stat(a));
elseif isequal(ud.category(a),2)
    % Do Counts
    cmin = vs.markers(ud.marker(a)).Counts.cmin(:,ud.stat(a));
    cmax = vs.markers(ud.marker(a)).Counts.cmax(:,ud.stat(a));
        
elseif isequal(ud.category(a),3)
    % Do Density
    cmin = vs.markers(ud.marker(a)).Density.cmin(:,ud.stat(a));
    cmax = vs.markers(ud.marker(a)).Density.cmax(:,ud.stat(a));
    
elseif isequal(ud.category(a),4)
    % Do Voxel
    cmin = vs.markers(ud.marker(a)).Counts.cmin(:,ud.stat(a));
    cmax = vs.markers(ud.marker(a)).Counts.cmax(:,ud.stat(a));
end

if nargout<3
    return
end

if isequal(ud.category(a),1)
    % Do volume
    values = vs.markers(ud.marker(a)).Volumes.values(:,ud.stat(a));
    title = vs.markers(ud.marker(a)).Volumes.title(ud.stat(a));
    name = vs.markers(ud.marker(a)).Volumes.name;
elseif isequal(ud.category(a),2)
    % Do Counts
    values = vs.markers(ud.marker(a)).Counts.values(:,ud.stat(a));
    title = vs.markers(ud.marker(a)).Counts.title(ud.stat(a));
    name = vs.markers(ud.marker(a)).Counts.name;
elseif isequal(ud.category(a),3)
    % Do Density
    values = vs.markers(ud.marker(a)).Density.values(:,ud.stat(a));
    title = vs.markers(ud.marker(a)).Density.title(ud.stat(a));   
    name = vs.markers(ud.marker(a)).Density.name;
elseif isequal(ud.category(a),4)
    % Do Density
    values = vs.markers(ud.marker(a)).Counts.values(:,ud.stat(a));
    title = vs.markers(ud.marker(a)).Counts.title(ud.stat(a));   
    name = vs.markers(ud.marker(a)).Counts.name;
end

end

function p = add_colorbar(p,vs,ud)

% Set colorbar
c = colorbar(p);

if ud.pos == 1
    % Set colorbar positions
    c.Position(1) = 0.0550;
    c.Position(3) = 0.015;
else
    % Set colorbar positions
    c.Position(1) = 0.93;
    c.Position(3) = 0.015;
end
    
% If annotation volume, add tick labels with acronyms for major areas
if ud.marker(ud.pos) == 0
    if isequal(ud.alpha,'false')
        % Set colorbar range
        c.Limits = [ud.cmin(ud.pos) - ud.adj(ud.pos) ud.cmax(ud.pos) + ud.adj(ud.pos)];

        c.Ticks = vs.av_cmap_pos;
        c.TickLabels = vs.av_cmap_labels;
    else
        % Set colorbar range
        idxs = vs.indexes(ismember(vs.indexes,vs.av_idx{ud.z}));
        c.Limits = [2, length(vs.av_stat_idx{ud.z})+1];
        c.Ticks = (2:length(vs.av_stat_idx{ud.z}))+0.5;
        acr = {vs.info.acronym};
        c.TickLabels = acr(idxs);
    end
else
    c.Limits = [ud.cmin(ud.pos)*0.99, min(6,ud.cmax(ud.pos))];
end

end

function fh_wbmfcn(f, vs)
% WindowButtonMotionFcn for the figure.
ud = get(f, 'UserData');

midline = round(f.Position(3)/2);
% The current point w.r.t the axis.
currPoint = get(f,'currentpoint');
if currPoint(1,1) < midline; p = 4; else; p = 2; end

currPoint = f.Children(p).CurrentPoint(1,1:2);
cx = round(currPoint(1)); cy = round(currPoint(2));

if cx > 0 && cy > 0 && cy < size(vs.av,1)
    idx = vs.av(cy,cx,ud.z);
    if idx > 0
        name = vs.info(idx).name;
    else
        idx = []; name = [];
    end
else
    idx = []; name = [];
end

% Add structure info top left
if ~isempty(name)
    set(ud.stext, 'String', name);
else
    set(ud.stext, 'String', 'not found');    
end

% Add values to left and right text boxes
if ~isempty(idx) && ismember(idx,vs.indexes)
   idx2 = find(vs.indexes == idx);
   if ud.marker(1) ~= 0 
       if ud.category(2) == 1 % Volume
         set(ud.p2text, 'String', vs.markers(ud.marker(1)).Volumes.values(idx2,ud.stat(1)))
        elseif ud.category(2) == 2 % Count
         set(ud.p2text, 'String', vs.markers(ud.marker(1)).Counts.values(idx2,ud.stat(1)))
        elseif ud.category(2) == 3 % Density
        set(ud.p2text, 'String', vs.markers(ud.marker(1)).Density.values(idx2,ud.stat(1)))
        end
   end
   if ud.marker(2) ~= 0
       if ud.category(2) == 1 % Volume
         set(ud.p2text, 'String', vs.markers(ud.marker(2)).Volumes.values(idx2,ud.stat(2)))
       elseif ud.category(2) == 2 % Count
           set(ud.p2text, 'String', vs.markers(ud.marker(2)).Counts.values(idx2,ud.stat(2)))
       elseif ud.category(2) == 3 % Density
           set(ud.p2text, 'String', vs.markers(ud.marker(2)).Density.values(idx2,ud.stat(2)))
       end
   end
else
    set(ud.p2text, 'String', 'NaN')
end

set(f, 'UserData', ud);

end

function p = sanitize_plot(p, idx)

set(p, 'Units', 'normalized');
set(p,'xtick',[])
set(p,'ytick',[])

set(p,'XColor','none')
set(p,'YColor','none')

daspect([1 1 1])

if idx == 1
    set(p, 'Position', [0.075, 0.025, 0.425, 0.85]);
else
    set(p, 'XDir','reverse')
    set(p, 'Position', [0.5, 0.025, 0.425, 0.85]);
end

end

