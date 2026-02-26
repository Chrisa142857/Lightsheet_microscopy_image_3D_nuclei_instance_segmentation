function [scat] = make_plot(df2,s2,disp_channel,sample_freq,colors)

marker_size = 0.1;
%colors = [0.4940 0.1840 0.5560];
%colors = [0.4660 0.6740 0.1880];
colors = [1, 0, 0];

if nargin<5
    colors = [];
end

disp(sum(df2(:,end) == disp_channel)/size(df2,1))

df3 = df2(any(df2(:,end) == disp_channel,2),:);
n_samples = round(size(df3,1)*sample_freq);
idx = randsample(size(df3,1),n_samples);
df3 = df3(idx,:);
%df3(:,1:3) = round(df3(:,1:3),1);
%df3 = bin_point_alpha(df2,disp_channel);

fig = figure('visible','off', 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
hold on
p2 = patch(s2,...
    'FaceColor',[0 0 0],...
    'EdgeColor','none',...
    'FaceLighting','flat');

%isonormals(BW,p2)
p2.FaceAlpha = 0.1;

scat = scatter3(df3(:,2),df3(:,1),df3(:,3),1,'.',...
    'MarkerEdgeAlpha', marker_size, 'MarkerFaceAlpha', marker_size);


%scat = plot3(df3(:,2),df3(:,1),df3(:,3),'o','MarkerSize',0.5);

if isempty(colors)
    scat = set_point_color(scat, disp_channel);
else
    scat = set_point_color(scat, disp_channel,colors);
end

%material 'dull'
axis vis3d off
daspect([1 1 1])
view([80,-60])
camroll(45)
hold off


%drawnow
%get marker handles
%   markers=scat.MarkerHandle;
%change transparency by altering 4th element
%    markers.EdgeColorType = 'truecoloralpha';
%    markers.EdgeColorData=uint8(255*[0.4940 0.1840 0.5565 0.1])';
exportgraphics(fig,'test.png','BackgroundColor','white','ContentType','image', 'Resolution',150)

end

function scat = set_point_color(scat, disp_channel,colors)

if nargin==3
    scat.MarkerEdgeColor = colors;
    scat.MarkerEdgeColor = colors;   
    return
end

switch disp_channel
    case 1
        scat.MarkerEdgeColor = [1 0 0];
        scat.MarkerEdgeColor = [1 0 0];  
    case 2
        scat.MarkerEdgeColor = [0.4660 0.6740 0.1880];
        scat.MarkerEdgeColor = [0.4660 0.6740 0.1880];               
    case 3
        scat.MarkerEdgeColor = [0.4940 0.1840 0.5560];
        scat.MarkerEdgeColor = [0.4940 0.1840 0.5560];
    case 4
        scat.MarkerEdgeColor = [0.4940 0.1840 0.5560];
        scat.MarkerEdgeColor = [0.4940 0.1840 0.5560];
end
end

function df_new = bin_point_alpha(df,idx)
df = df(df(:,end) == idx,:);

df4 = df;
pos = round(df4(:,1:3));

[a,b,c] = unique(pos,'rows');

[cc,~,bins] = histcounts(c,'BinMethod','integers');
cc = cc/max(cc);

alpha = zeros(size(pos,1),1);
alpha = cc(bins);

alphas(c) = alpha;
pos = a;

df_new = horzcat(pos,alphas',repmat(idx,length(alphas),1));

end