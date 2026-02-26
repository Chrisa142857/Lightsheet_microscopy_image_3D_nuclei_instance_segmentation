function [p1,p2] = display_slice2(vs,key)
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
%f.Position = [5,600,855,600]; %AR = 1.425
f.Position = [5,600,741,520];
f.Position = [5,600,1710*1.1,1200*1.1];


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
set(f, 'WindowScrollWheelFcn', @(src,evt)updateSlice(f, evt, vs)) % Set wheel detector. 
set(f, 'KeyPressFcn', @(f,k)hotkeyFcn(f, k, vs)); % Hot keys
f.PaperOrientation = 'landscape';

display_help
end

function hotkeyFcn(f, keydata, vs)

ud = get(f, 'UserData');

switch lower(keydata.Key)    
    case 'h' % display help
        display_help
        return
    
    case 'l' % swtich to left plot
        fprintf('Left plot selected \n');
        ud.pos = 1;
        set(f,'CurrentAxes',f.Children(4))

    case 'r' % switch to right plot
        fprintf('Right plot selected \n');
        ud.pos = 2;
        set(f,'CurrentAxes',f.Children(2))
        
    case 'b' % toggle borders
        if ud.pos == 1
            state =  f.Children(4).Children(1).Visible;
            if isequal(state,'on')
                f.Children(4).Children(1).Visible = 'off';
            else
                f.Children(4).Children(1).Visible = 'on';
            end
        else
            state =  f.Children(2).Children(1).Visible;
            if isequal(state,'on')
                f.Children(2).Children(1).Visible = 'off';
            else
                f.Children(2).Children(1).Visible = 'on';
            end
        end
    case 'a' % toggle alpha
        if ud.pos == 1
            state =  f.Children(4).Children(2).Visible;
            if isequal(state,'on')
                f.Children(4).Children(2).Visible = 'off';
            else
                f.Children(4).Children(2).Visible = 'on';
            end
        else
            state =  f.Children(2).Children(2).Visible;
            if isequal(state,'on')
                f.Children(2).Children(2).Visible = 'off';
            else
                f.Children(2).Children(2).Visible = 'on';
            end
        end        
    case 'm' % switch marker
        prompt = sprintf('Select marker number (1-%d): ',length(vs.markers));
        input1 = input(prompt,'s');
    
        new_marker = str2double(input1);
        ud.marker(ud.pos) = new_marker;
        
        if ud.category(ud.pos) == 1 && new_marker > 1
            fprintf('Volume data displayed. Marker change has no effect\n');
            ud.marker(ud.pos) = 1;
            shg
            return
        elseif new_marker > length(vs.markers)
            fprintf('Marker number out of bounds\n');
            return
        end
        
        if ud.pos == 1
            [p1,cmin,cmax] = display_plot(ud,vs,f);
            p1 = sanitize_plot(p1,1);
            ud.cmin(1) = cmin; ud.cmax(1) = cmax;
            
            % Set colorbar
            add_colorbar(p1,vs,ud)
        else
            [p2,cmin,cmax] = display_plot(ud,vs,f);
            p2 = sanitize_plot(p2,2);
            ud.cmin(2) = cmin; ud.cmax(2) = cmax;

            % Set colorbar
            add_colorbar(p2,vs,ud);
        end
        
        if new_marker == 0
            fprintf('Displaying annotation volume');
        elseif new_marker>length(vs.marker_names)
            fprintf('Displaying custom calculation %d\n',new_marker-length(vs.marker_names));            
        else
            fprintf('Displaying marker %s\n',vs.marker_names(new_marker));
        end

    case 'c' % switch stats category
        if ud.marker(ud.pos) == 0
            fprintf('Annotation Volume Selected. Switch Plots \n');
           return 
        end
        
        input1 = input('Select stat category (v/c/d): ','s');
    
        switch input1
            case 'v'
                if ~isfield(vs.markers(ud.marker(ud.pos)),'Volumes')
                    fprintf('Volume data not present \n');
                    return
                else 
                    new_cat = 1;
                    prompt = "volume data";
                end
            case 'c'
                if ~isfield(vs.markers(ud.marker(ud.pos)),'Counts')
                    fprintf('Count data not present \n');
                    return
                else
                    new_cat = 2;
                    prompt = "count data";
                end
            case 'd'
                if ~isfield(vs.markers(ud.marker(ud.pos)),'Density')
                    fprintf('Density data not present \n');
                    return
                else
                    new_cat = 3;
                    prompt = "Density data";
                end
            otherwise
                fprintf('Wrong category selection \n');
                return
        end            
        
        ud.category(ud.pos) = new_cat;
        
        if new_cat == 1
            ud.marker(ud.pos) = 1;
        end
        
        if ud.pos == 1
            [p1,cmin,cmax] = display_plot(ud,vs,f);
            p1 = sanitize_plot(p1,1);
            ud.cmin(1) = cmin; ud.cmax(1) = cmax;
            
            % Set colorbar
            add_colorbar(p1,vs,ud)
        else
            [p2,cmin,cmax] = display_plot(ud,vs,f);
            p2 = sanitize_plot(p2,2);
            ud.cmin(2) = cmin; ud.cmax(2) = cmax;

            % Set colorbar
            add_colorbar(p2,vs,ud);
        end
        
        fprintf('Displaying %s\n',prompt);
        
    case 's' % switch stats values
        if ud.marker(ud.pos) == 0
            fprintf('Annotation Volume Selected. Switch Plot \n');
            return 
        end
        
        input1 = input('Select statistic (1-8): ','s');
        
        new_stat = str2double(input1);
        ud.stat(ud.pos) = new_stat;
                
        if ud.pos == 1
            [p1,cmin,cmax] = display_plot(ud,vs,f);
            p1 = sanitize_plot(p1,1);
            ud.cmin(1) = cmin; ud.cmax(1) = cmax;
            
            % Set colorbar
            add_colorbar(p1,vs,ud)
        else
            [p2,cmin,cmax] = display_plot(ud,vs,f);
            p2 = sanitize_plot(p2,2);
            ud.cmin(2) = cmin; ud.cmax(2) = cmax;

            % Set colorbar
            add_colorbar(p2,vs,ud);
        end
        
        [~,~,prompt] = retrieve_data(vs,ud);        
        fprintf('Displaying %s\n',prompt);
end

set(f, 'UserData', ud);

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
    
    if ud.pos == 1
       cmin = -100*1.02;cmax=100*1.02; 
    else
        cmin = 1*1.02; cmax = 5*1.02;
    end

    
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
    title = vs.markers(ud.marker(a)).Counts.title(ud.stat(a));

elseif isequal(ud.category(a),3)
    %  Do Density
    stats = vs.markers(ud.marker(a)).Density.values(:,ud.stat(a));
    title = vs.markers(ud.marker(a)).Density.title(ud.stat(a));

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
    c.Limits = [ud.cmin(ud.pos)*0.99 ud.cmax(ud.pos)*0.99];
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

function updateSlice(f, evt, vs)

ud = get(f, 'UserData');
pre_ud_pos = ud.pos;


% scroll through slices
ud.z = ud.z - evt.VerticalScrollCount;

if ud.z>size(vs.av,3); ud.z = 1; end %wrap around
if ud.z<1; ud.z = size(vs.av,3); end %wrap around

vs.z = ud.z;

% Update image p1
ud.pos = 1;
[slice,~,~] = retrieve_slice(ud,vs);
%slice(~vs.binMask(:,:,ud.z)) = ud.cmin(ud.pos) - ud.adj(ud.pos);
f.Children(4).Children(3).CData = single(slice);

% Update alpha p1
%a = ~ismember(single(vs.av(:,:,ud.z)),vs.indexes);
%f.Children(4).Children(2).CData = a;

%if ud.marker(ud.pos) ~= 0
%    f.Children(4).Children(2).AlphaData = a;
%else
%    f.Children(4).Children(2).AlphaData = a*ud.alpha_val;
%end

% Overlay boundaries p1
boundaries = vs.boundaries(:,:,ud.z)*max(f.Children(4).Children(1).CData,[],'all');
f.Children(4).Children(1).CData = boundaries;
f.Children(4).Children(1).AlphaData = vs.boundaries(:,:,ud.z);

% Update image p2
ud.pos = 2;
[slice,~,~] = retrieve_slice(ud,vs);
%slice(~vs.binMask(:,:,ud.z)) = ud.cmin(ud.pos) - ud.adj(ud.pos);
f.Children(2).Children(3).CData = single(slice);

% Update alpha p2
%a = ~ismember(single(vs.av(:,:,ud.z)),vs.indexes);
%f.Children(2).Children(2).CData = a;
%if ud.marker(ud.pos) ~= 0
%    f.Children(2).Children(2).AlphaData = a;
%else
%    f.Children(2).Children(2).AlphaData = a*ud.alpha_val;
%end

% Overlay boundaries p2
boundaries = vs.boundaries(:,:,ud.z)*max(f.Children(2).Children(1).CData,[],'all');
f.Children(2).Children(1).CData = boundaries;
f.Children(2).Children(1).AlphaData = vs.boundaries(:,:,ud.z);

% Update AP coordinates
ud.ptext.String = sprintf('Slice:%d/%d \t %.2f AP',ud.z,size(vs.av,3),vs.ap(ud.z));

ud.pos = pre_ud_pos;

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

function display_help

fprintf(1, '\nControls: \n');
fprintf(1, '--------- \n');
fprintf(1, 'scroll: move between slices \n');
fprintf(1, 'click: print structure data \n');
fprintf(1, 'l/r: toggle left(l) or right(r) plot to update \n');
fprintf(1, 'b: toggle viewing boundaries \n');
fprintf(1, 'a: toggle highlight regions \n');
fprintf(1, 'm: switch marker \n');
fprintf(1, 'c: switch data type (i.e. volume, Counts, Density) \n');
fprintf(1, 's: switch data (i.e. mean, st. dev., %% change, p value) \n');

end
