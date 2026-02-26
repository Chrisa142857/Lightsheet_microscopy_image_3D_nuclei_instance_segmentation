function [p1,p2] = display_slice3(vs,key)
% User defined keys
% z: z_position
% marker: which marker
% category: counts, volume, density
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
ud.alpha_val = 1;
ud.init = 'true';
ud.imgL = zeros(size(vs.av(:,:,1)));
ud.imgR = zeros(size(vs.av(:,:,1)));

f = figure;
f.Position = [5,600,855,600]; %AR = 1.425

[p1,cmin,cmax] = display_plot(ud,vs,f);
ud.cmin(1) = cmin; ud.cmax(1) = cmax;
p1 = sanitize_plot(p1,1);

ud.pos = 2;
[p2,cmin,cmax] = display_plot(ud,vs,f);
ud.cmin(2) = cmin; ud.cmax(2) = cmax;
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

fprintf(1, '\nControls: \n');
fprintf(1, '--------- \n');
fprintf(1, 'scroll: move between slices \n');
fprintf(1, 'click: print structure data \n');
fprintf(1, 'l/r: toggle left(l) or right(r) plot to update \n');
fprintf(1, 'b: toggle viewing boundaries \n');
fprintf(1, 'a: toggle highlight regions \n');
fprintf(1, 'm: switch marker \n');
fprintf(1, 'c: switch data type (i.e. volume, counts, density) \n');
fprintf(1, 's: switch data (i.e. mean, st. dev., %% change, p value) \n');

ud.init = 'false';
set(f, 'UserData', ud);
set(f, 'WindowButtonDownFcn',@(f,k)fh_wbmfcn(f, vs)) % Set click detector.
set(f, 'WindowScrollWheelFcn', @(src,evt)updateSlice(f, evt, vs)) % Set wheel detector. 
set(f, 'KeyPressFcn', @(f,k)hotkeyFcn(f, k, vs)); % Hot keys
end

function [p,cmin,cmax] = display_plot(ud,vs,f)
% Initial plot display

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
h1 = imagesc(single(slice));

% Set caxis mappings
if ud.marker(ud.pos) == 0
    cmin = 0; cmax = length(colors)+1;
    h1.CDataMapping = 'direct';
    caxis([cmin cmax])
else
    [~,~,~,cmin,cmax] = retrieve_data(vs,ud);
    h1.CDataMapping = 'scaled';    
    if cmin<0
        %cmin = -25;
        %cmax = 0;
        %colors = colors(1:123,:);
        
        %h1.CData = round(h1.CData);        
        %cmin = -abs(cmax);        
    else
        %cmin = -1;
        cmax = cmax;
        cmin = -cmax/254;
    end
    caxis([cmin cmax])
end

% Add title
p.Title.String = title;
p.Title.FontSize = 14;

% Overlay boundaries
hold on

% Add alpha to annotation volume
a = ~ismember(single(vs.av(:,:,ud.z)),single(vs.indexes));
h3 = image(a);
h1.CDataMapping = 'scaled'; 

if ud.marker(ud.pos) == 0
    set(h3,'AlphaData',a*ud.alpha_val)
else
    set(h3,'AlphaData',a)
end

h3.Visible = 'off';
%if ud.marker(ud.pos) ~= 0
%    h3.Visible  = 'on';
%elseif ~isequal(ud.alpha,'true')    
%    h3.Visible  = 'off';
%end

boundaries = vs.boundaries(:,:,ud.z)*-1;
h2 = image(boundaries);
boundaries = boundaries<0;
set(h2,'AlphaData',boundaries)
hold off

colormap(p,[[0.25,0.25,0.25];colors;])
end

function [title,name,values,cmin,cmax] = retrieve_data(vs,ud)

% Get axis
a = ud.pos;

if isequal(ud.category(a),1)
    % Do volume
    values = vs.markers(ud.marker(a)).volumes.values(:,ud.stat(a));
    title = vs.markers(ud.marker(a)).volumes.title(ud.stat(a));
    name = vs.markers(ud.marker(a)).volumes.name;
    cmin = vs.markers(ud.marker(a)).volumes.cmin(:,ud.stat(a));
    cmax = vs.markers(ud.marker(a)).volumes.cmax(:,ud.stat(a));
elseif isequal(ud.category(a),2)
    % Do counts
    values = vs.markers(ud.marker(a)).counts.values(:,ud.stat(a));
    title = vs.markers(ud.marker(a)).counts.title(ud.stat(a));
    name = vs.markers(ud.marker(a)).counts.name;
    cmin = vs.markers(ud.marker(a)).counts.cmin(:,ud.stat(a));
    cmax = vs.markers(ud.marker(a)).counts.cmax(:,ud.stat(a));
        
elseif isequal(ud.category(a),3)
    % Do density
    values = vs.markers(ud.marker(a)).density.values(:,ud.stat(a));
    title = vs.markers(ud.marker(a)).density.title(ud.stat(a));   
    name = vs.markers(ud.marker(a)).density.name;
    cmin = vs.markers(ud.marker(a)).density.cmin(:,ud.stat(a));
    cmax = vs.markers(ud.marker(a)).density.cmax(:,ud.stat(a));
end

end



function [slice,colors,title] = retrieve_slice(ud,vs)

% Get which axis
a = ud.pos;

% Displaying annotation volume
if ud.marker(a) == 0
    slice = vs.av(:,:,ud.z);
    colors = vs.av_colors;
    
    s1 = 1:length(vs.info);
    s1 = s1(ismember(s1,vs.indexes));
    s1 = s1(ismember(s1,vs.av_idx{ud.z}));
    
    acronyms = retrieve_acronyms(vs,s1);
    ud.acronyms = acronyms;
    colors = [1,1,1;colors(s1,:)];
    
    slice(~ismember(slice,s1)) = 1;
    
    [~, ~,i3] = unique(slice);
    s1 = [1,s1];
    s1 = 1:length(s1);
    slice2 = s1(i3);
    slice = reshape(slice2, size(slice));    
    
    
    a = 1;
    
    
    %c1 = vs.av_colors(s1,:);
    %img = vs.av(:,:,ud.z);
    %img(~ismember(img,s1)) = 0;
    
    
    %colormap(c1)
    
    
    
    
    
    title = "Annotation Volume";
else
    % Displaying some statistics
    slice = double(vs.av(:,:,ud.z));
    idxs = vs.av_idx{ud.z};
    if isequal(ud.category(a),1)
        % Do volume
        stats = vs.markers(ud.marker(a)).volumes.values(:,ud.stat(a));
        title = vs.markers(ud.marker(a)).volumes.title(ud.stat(a));
        
    elseif isequal(ud.category(a),2)
        % Do counts
        stats = vs.markers(ud.marker(a)).counts.values(:,ud.stat(a));
        title = vs.markers(ud.marker(a)).counts.title(ud.stat(a));
        
    elseif isequal(ud.category(a),3)
        %  Do density
        stats = vs.markers(ud.marker(a)).density.values(:,ud.stat(a));
        title = vs.markers(ud.marker(a)).density.title(ud.stat(a));
        
    else
        error("Invalid stats category selected")
    end

    % Recolor slice according to stats
    for i = 1:length(idxs)
        slice(slice == idxs(i)) = stats(idxs(i));
    end
    
    % Get colormap
    colors = vs.markers(ud.marker(a)).colors{ud.stat(a)};
end

end

function p = add_colorbar(p,vs,ud)

if isequal(ud.init,'false')
    if ud.marker(ud.pos) == 0
        update_annotation_colorbar(p,vs,ud);
    end
    
elseif ud.pos == 1
% Set colorbar
c = colorbar(p);
c.Limits = [ud.cmin(1) ud.cmax(1)];

% Set colorbar positions
c.Position(1) = 0.0550;
c.Position(3) = 0.015;

    % If annotation volume, add tick labels with acronyms for major areas
    if ud.marker(1) == 0
        c.Ticks = vs.av_cmap_pos;
        %c.TickLabels = vs.av_cmap_labels;

        c.TickLabels = [];
        c.Limits = [1 ud.cmax(1)];
        labels = {vs.info.acronym};

        
        idx = vs.av_idx{ud.z};
        idx = idx(vs.bin_idx{ud.z});

        colors = vs.av_colors(idx,:);
        
        [~,b] = unique(colors,'rows');
        c.Ticks = sort(b+1);
        c.TickLabels = {vs.info(sort(idx(b)')).acronym};
        
       %%%%%
        %s1 = 1:length(vs.info);
        %s1 = s1(ismember(s1,vs.indexes));
        %s1 = s1(ismember(s1,vs.av_idx{ud.z}));
        %c1 = vs.av_colors(s1,:);
        %img = vs.av(:,:,ud.z);
        %img(~ismember(img,s1)) = 0;
        %colormap(c1)    
        %s2 = 1:length(s1);
        %colormap s2
        %c.Limits = [min(s2) max(s2)];

        %idxs = ismember(vs.indexes,vs.av_idx{ud.z});
        %idxs = vs.av_idx{ud.z}(idxs);
        %c.Ticks = 1:length(idxs);
        %c.Limits = [min(idxs) max(idxs)];  
        %c.Ticks = round((idxs(1:end-1)+idxs(2:end))/2);
        %c.Ticks = unique(idxs)-1;
    end 
end

if ud.pos == 2
    % Set colorbar
    c = colorbar(p);

    % Set colorbar positions
    c.Position(1) = 0.93;
    c.Position(3) = 0.015;
    c.Limits = [ud.cmin(2) ud.cmax(2)];

    % If annotation volume, add tick labels with acronyms for major areas
    if ud.marker(2) == 0
        c.Ticks = vs.av_cmap_pos;
        c.TickLabels = vs.av_cmap_labels;
    end
end

end

function update_annotation_colorbar(p,vs,ud)


idx = vs.av_idx{ud.z};
idx = idx(vs.bin_idx{ud.z});

colors = vs.av_colors(idx,:);

a = 1;







end


function hotkeyFcn(f, keydata, vs)

ud = get(f, 'UserData');

switch lower(keydata.Key)    
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
        disp(state)
        
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
        
        if new_marker > 0
            fprintf('Displaying marker %s\n',vs.marker_names(new_marker));
        else
            fprintf('Displaying annotation volume\n');
        end

    case 'c' % switch stats category
        if ud.marker(ud.pos) == 0
            fprintf('Annotation Volume Selected. Switch Plots \n');
           return 
        end
        
        input1 = input('Select stat category (v/c/d): ','s');
    
        switch input1
            case 'v'
                if ~isfield(vs.markers(ud.marker(ud.pos)),'volumes')
                    fprintf('Volume data not present \n');
                    return
                else 
                    new_cat = 1;
                    prompt = "volume data";
                end
            case 'c'
                if ~isfield(vs.markers(ud.marker(ud.pos)),'counts')
                    fprintf('Count data not present \n');
                    return
                else
                    new_cat = 2;
                    prompt = "count data";
                end
            case 'd'
                if ~isfield(vs.markers(ud.marker(ud.pos)),'density')
                    fprintf('Density data not present \n');
                    return
                else
                    new_cat = 3;
                    prompt = "density data";
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
        
        prompt = retrieve_data(vs,ud);        
        fprintf('Displaying %s\n',prompt);
end

set(f, 'UserData', ud);
gcf(f)

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
if ~isempty(idx)
   if ud.marker(1) ~= 0 
       if ud.category(2) == 1 % Volume
         set(ud.p2text, 'String', vs.markers(ud.marker(1)).volumes.values(idx,ud.stat(1)))
        elseif ud.category(2) == 2 % Count
         set(ud.p2text, 'String', vs.markers(ud.marker(1)).counts.values(idx,ud.stat(1)))
        elseif ud.category(2) == 3 % Density
        set(ud.p2text, 'String', vs.markers(ud.marker(1)).density.values(idx,ud.stat(1)))
        end
   end
   if ud.marker(2) ~= 0
       if ud.category(2) == 1 % Volume
         set(ud.p2text, 'String', vs.markers(ud.marker(2)).volumes.values(idx,ud.stat(2)))
       elseif ud.category(2) == 2 % Count
           set(ud.p2text, 'String', vs.markers(ud.marker(2)).counts.values(idx,ud.stat(2)))
       elseif ud.category(2) == 3 % Density
           set(ud.p2text, 'String', vs.markers(ud.marker(2)).density.values(idx,ud.stat(2)))
       end
    end
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
f.Children(4).Children(3).CData = single(slice);
add_colorbar(f.Children(1),vs,ud);

% Update alpha p1
a = ~ismember(single(vs.av(:,:,ud.z)),vs.indexes);
f.Children(4).Children(2).CData = a;

if ud.marker(ud.pos) ~= 0
    f.Children(4).Children(2).AlphaData = a;
else
    f.Children(4).Children(2).AlphaData = a*ud.alpha_val;
end

% Overlay boundaries p1
boundaries = vs.boundaries(:,:,ud.z)*-1;
f.Children(4).Children(1).CData = boundaries;
boundaries = boundaries<0;
f.Children(4).Children(1).AlphaData = boundaries;

% Update image p2
ud.pos = 2;
[slice,~,~] = retrieve_slice(ud,vs);
f.Children(2).Children(3).CData = single(slice);

% Update alpha p2
a = ~ismember(single(vs.av(:,:,ud.z)),vs.indexes);
f.Children(2).Children(2).CData = a;
if ud.marker(ud.pos) ~= 0
    f.Children(2).Children(2).AlphaData = a;
else
    f.Children(2).Children(2).AlphaData = a*ud.alpha_val;
end

% Overlay boundaries p2
boundaries = vs.boundaries(:,:,ud.z)*-1;
f.Children(2).Children(1).CData = boundaries;
boundaries= boundaries<0;
f.Children(2).Children(1).AlphaData = boundaries;


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

function acronyms = retrieve_acronyms(vs,indexes)

acronyms = string({vs.info.acronym});
acronyms = acronyms(indexes);

end

