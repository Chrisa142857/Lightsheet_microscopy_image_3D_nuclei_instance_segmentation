function map = brewermap(N,scheme)
% The complete selection of ColorBrewer colorschemes (RGB colormaps).
%
% (c) 2014 Stephen Cobeldick
%
% Returns any RGB colormap from the ColorBrewer colorschemes, especially
% intended for mapping and plots with attractive, distinguishable colors.
%
%%% Syntax (basic):
%  map = brewermap(N,scheme); % Select colormap length, select any colorscheme.
%  brewermap('plot')          % View a figure showing all ColorBrewer colorschemes.
%  schemes = brewermap('list')% Return a list of all ColorBrewer colorschemes.
%  [map,num,typ] = brewermap(...); % The current colorscheme's number of nodes and type.
%
%%% Syntax (preselect colorscheme):
%  old = brewermap(scheme); % Preselect any colorscheme, return the previous scheme.
%  map = brewermap(N);      % Use preselected scheme, select colormap length.
%  map = brewermap;         % Use preselected scheme, length same as current figure's colormap.
%
% See also CUBEHELIX RGBPLOT3 RGBPLOT COLORMAP COLORBAR PLOT PLOT3 SURF IMAGE AXES SET JET LBMAP PARULA
%
%% Color Schemes %%
%
% This product includes color specifications and designs developed by Cynthia Brewer.
% See the ColorBrewer website for further information about each colorscheme,
% colour-blind suitability, licensing, and citations: http://colorbrewer.org/
%
% To reverse the colormap sequence simply prefix the string token with '*'.
%
% Each colorscheme is defined by a set of hand-picked RGB values (nodes).
% If <N> is greater than the requested colorscheme's number of nodes then:
%  * Sequential and Diverging schemes are interpolated to give a larger
%    colormap. The interpolation is performed in the Lab colorspace.
%  * Qualitative schemes are repeated to give a larger colormap.
% Else:
%  * Exact values from the ColorBrewer sequences are returned for all colorschemes.
%
%%% Diverging
%
% Scheme|'BrBG'|'PRGn'|'PiYG'|'PuOr'|'RdBu'|'RdGy'|'RdYlBu'|'RdYlGn'|'Spectral'|
% ------|------|------|------|------|------|------|--------|--------|----------|
% Nodes |  11  |  11  |  11  |  11  |  11  |  11  |   11   |   11   |    11    |
%
%%% Qualitative
%
% Scheme|'Accent'|'Dark2'|'Paired'|'Pastel1'|'Pastel2'|'Set1'|'Set2'|'Set3'|
% ------|--------|-------|--------|---------|---------|------|------|------|
% Nodes |   8    |   8   |   12   |    9    |    8    |   9  |  8   |  12  |
%
%%% Sequential
%
% Scheme|'Blues'|'BuGn'|'BuPu'|'GnBu'|'Greens'|'Greys'|'OrRd'|'Oranges'|'PuBu'|
% ------|-------|------|------|------|--------|-------|------|---------|------|
% Nodes |   9   |  9   |  9   |  9   |   9    |   9   |  9   |    9    |  9   |
%
% Scheme|'PuBuGn'|'PuRd'|'Purples'|'RdPu'|'Reds'|'YlGn'|'YlGnBu'|'YlOrBr'|'YlOrRd'|
% ------|--------|------|---------|------|------|------|--------|--------|--------|
% Nodes |   9    |  9   |    9    |  9   |  9   |  9   |   9    |   9    |   9    |
%
%% Examples %%
%
%%% Plot a scheme's RGB values:
% rgbplot(brewermap(9,'Blues'))  % standard
% rgbplot(brewermap(9,'*Blues')) % reversed
%
%%% View information about a colorscheme:
% [~,num,typ] = brewermap(0,'Paired')
% num = 12
% typ = 'Qualitative'
%
%%% Multi-line plot using matrices:
% N = 6;
% axes('ColorOrder',brewermap(N,'Pastel2'),'NextPlot','replacechildren')
% X = linspace(0,pi*3,1000);
% Y = bsxfun(@(x,n)n*sin(x+2*n*pi/N), X(:), 1:N);
% plot(X,Y, 'linewidth',4)
%
%%% Multi-line plot in a loop:
% set(0,'DefaultAxesColorOrder',brewermap(NaN,'Accent'))
% N = 6;
% X = linspace(0,pi*3,1000);
% Y = bsxfun(@(x,n)n*sin(x+2*n*pi/N), X(:), 1:N);
% for n = 1:N
%     plot(X(:),Y(:,n), 'linewidth',4);
%     hold all
% end
%
%%% New colors for the COLORMAP example:
% load spine
% image(X)
% colormap(brewermap([],'YlGnBu'))
%
%%% New colors for the SURF example:
% [X,Y,Z] = peaks(30);
% surfc(X,Y,Z)
% colormap(brewermap([],'RdYlGn'))
% axis([-3,3,-3,3,-10,5])
%
%%% New colors for the CONTOURCMAP example:
% brewermap('PuOr'); % preselect the colorscheme.
% load topo
% load coast
% figure
% worldmap(topo, topolegend)
% contourfm(topo, topolegend);
% contourcmap('brewermap', 'Colorbar','on', 'Location','horizontal',...
% 'TitleString','Contour Intervals in Meters');
% plotm(lat, long, 'k')
%
%% Input and Output Arguments %%
%
%%% Inputs (*=default):
% N = NumericScalar, N>=0, an integer to specify the colormap length.
%   = *[], same length as the current figure's colormap (see COLORMAP).
%   = NaN, same length as the defining RGB nodes (useful for Line ColorOrder).
%   = CharRowVector, to preselect a ColorBrewer colorscheme for later use.
%   = 'plot', create a figure showing all of the ColorBrewer colorschemes.
%   = 'list', return a cell array of strings listing all ColorBrewer colorschemes.
% scheme = CharRowVector, a ColorBrewer colorscheme name.
%        = *none, uses the preselected colorscheme (must be set previously!).
%
%%% Outputs:
% map = NumericMatrix, size Nx3, a colormap of RGB values between 0 and 1.
% num = NumericScalar, the number of nodes defining the ColorBrewer colorscheme.
% typ = CharRowVector, the colorscheme type: 'Diverging'/'Qualitative'/'Sequential'.
% OR
% schemes = CellOfCharRowVectors, a list of every ColorBrewer colorscheme.
%
% [map,num,typ] = brewermap(*N,*scheme)
% OR
% schemes = brewermap('list')


%% Added
if contains(scheme,'viridis')
    map=viridis(N);
    if isequal(scheme,'*viridis')
        map = flip(map,1);
    end
    return
elseif contains(scheme,'plasma')
    map=plasma(N);
    if isequal(scheme,'*plasma')
        map = flip(map,1);
    end
    return
end




%% Input Wrangling %%
%
persistent raw tok isr idp
%
if isempty(raw)
	raw = bmColors();
end
%
msg = 'A colorscheme must be preselected before calling without a colorscheme name.';
%
if nargin==0 % Current figure's colormap length and the preselected colorscheme.
	assert(~isempty(idp),msg)
	[map,num,typ] = bmSample([],isr,raw(idp));
elseif nargin==2 % Input colormap length and colorscheme.
	assert(isnumeric(N),'The first argument must be a scalar numeric, or empty.')
	assert(ischar(scheme)&&isrow(scheme),'The second argument must be a 1xN char.')
	tmp = strncmp('*',scheme,1);
	[map,num,typ] = bmSample(N,tmp,raw(bmMatch(scheme,tmp,raw)));
elseif isnumeric(N) % Input colormap length and the preselected colorscheme.
	assert(~isempty(idp),msg)
	[map,num,typ] = bmSample(N,isr,raw(idp));
else
	assert(ischar(N)&&isrow(N),'The first argument must be a 1xN char or a scalar numeric.')
	switch lower(N)
		case 'plot' % Plot all colorschemes in a figure.
			bmPlotFig(raw)
		case 'list' % Return a list of all colorschemes.
			map = {raw.str};
			typ = {raw.typ};
			num = [raw.num];
		otherwise % Store the preselected colorscheme token.
			tmp = strncmp('*',N,1);
			idp = bmMatch(N,tmp,raw);
			typ = raw(idp).typ;
			num = raw(idp).num;
			% Only update persistent values if colorscheme name is okay:
			isr = tmp;
			map = tok;
			tok = N;
	end
end
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%brewermap
function idx = bmMatch(str,tmp,raw)
% Match the requested colorscheme name to names in the raw data structure.
str = str(1+tmp:end);
idx = strcmpi({raw.str},str);
assert(any(idx),'Colorscheme "%s" is not supported. Check the colorscheme list.',str)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%bmMatch
function [map,num,typ] = bmSample(N,isr,raw)
% Pick a colorscheme, downsample/interpolate to the requested colormap length.
%
num = raw.num;
typ = raw.typ;
%
if isempty(N)
	N = size(get(gcf,'colormap'),1);
elseif isscalar(N)&&isnan(N)
	N = num;
else
	assert(isscalar(N),'First argument must be a numeric scalar, or empty.')
	assert(isreal(N),'Input <N> must be a real numeric: %g+%gi',N,imag(N))
	assert(fix(N)==N&&N>=0,'Input <N> must be positive integer: %g',N)
end
%
if N==0
	map = nan(0,3);
	return
end
%
% downsample:
[idx,itp] = bmIndex(N,num,typ);
map = raw.rgb(idx,:)/255;
% interpolate:
if itp
	M = [...
		+3.2406255,-1.5372080,-0.4986286;...
		-0.9689307,+1.8757561,+0.0415175;...
		+0.0557101,-0.2040211,+1.0569959];
	wpt = [0.95047,1,1.08883]; % D65
	%
	map = bmRGB2Lab(map,M,wpt); % optional
	%
	% Extrapolate a small amount at both ends:
	%vec = linspace(0,num+1,N+2);
	%map = interp1(1:num,map,vec(2:end-1),'linear','extrap');
	% Interpolation completely within ends:
	map = interp1(1:num,map,linspace(1,num,N),'spline');
	%
	map = bmLab2RGB(map,M,wpt); % optional
end
% reverse order:
if isr
	map = map(end:-1:1,:);
end
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%bmSample
function rgb = bmGammaCor(rgb)
% Gamma correction of sRGB data.
idx = rgb <= 0.0031308;
rgb(idx) = 12.92 * rgb(idx);
rgb(~idx) = real(1.055 * rgb(~idx).^(1/2.4) - 0.055);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%bmGammaCor
function rgb = bmGammaInv(rgb)
% Inverse gamma correction of sRGB data.
idx = rgb <= 0.04045;
rgb(idx) = rgb(idx) / 12.92;
rgb(~idx) = real(((rgb(~idx) + 0.055) / 1.055).^2.4);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%bmGammaInv
function lab = bmRGB2Lab(rgb,M,wpt) % Nx3 <- Nx3
% Convert a matrix of sRGB values to Lab.
%
%applycform(rgb,makecform('srgb2lab','AdaptedWhitePoint',wpt))
%
% RGB2XYZ:
xyz = bmGammaInv(rgb) / M.';
% Remember to include my license when copying my implementation.
% XYZ2Lab:
xyz = bsxfun(@rdivide,xyz,wpt);
idx = xyz>(6/29)^3;
F = idx.*(xyz.^(1/3)) + ~idx.*(xyz*(29/6)^2/3+4/29);
lab(:,2:3) = bsxfun(@times,[500,200],F(:,1:2)-F(:,2:3));
lab(:,1) = 116*F(:,2) - 16;
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%bmRGB2Lab
function rgb = bmLab2RGB(lab,M,wpt) % Nx3 <- Nx3
% Convert a matrix of Lab values to sRGB.
%
%applycform(lab,makecform('lab2srgb','AdaptedWhitePoint',wpt))
%
% Lab2XYZ
tmp = bsxfun(@rdivide,lab(:,[2,1,3]),[500,Inf,-200]);
tmp = bsxfun(@plus,tmp,(lab(:,1)+16)/116);
idx = tmp>(6/29);
tmp = idx.*(tmp.^3) + ~idx.*(3*(6/29)^2*(tmp-4/29));
xyz = bsxfun(@times,tmp,wpt);
% Remember to include my license when copying my implementation.
% XYZ2RGB
rgb = max(0,min(1, bmGammaCor(xyz * M.')));
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%cbLab2RGB
function bmPlotFig(raw)
% Creates a figure showing all of the ColorBrewer colorschemes.
%
persistent cbh axh
%
xmx = max([raw.num]);
ymx = numel(raw);
%
if ishghandle(cbh)
	figure(cbh);
	delete(axh);
else
	cbh = figure('HandleVisibility','callback', 'IntegerHandle','off',...
		'NumberTitle','off', 'Name',[mfilename,' Plot'],'Color','white',...
		'MenuBar','figure', 'Toolbar','none', 'Tag',mfilename);
	set(cbh,'Units','pixels')
	pos = get(cbh,'Position');
	pos(1:2) = pos(1:2) - 123;
	pos(3:4) = max(pos(3:4),[842,532]);
	set(cbh,'Position',pos)
end
%
axh = axes('Parent',cbh, 'Color','none',...
	'XTick',0:xmx, 'YTick',0.5:ymx, 'YTickLabel',{raw.str}, 'YDir','reverse');
title(axh,['ColorBrewer Color Schemes (',mfilename,'.m)'], 'Interpreter','none')
xlabel(axh,'Scheme Nodes')
ylabel(axh,'Scheme Name')
axf = get(axh,'FontName');
%
for y = 1:ymx
	num = raw(y).num;
	typ = raw(y).typ;
	map = raw(y).rgb(bmIndex(num,num,typ),:)/255; % downsample
	for x = 1:num
		patch([x-1,x-1,x,x],[y-1,y,y,y-1],1, 'FaceColor',map(x,:), 'Parent',axh)
	end
	text(xmx+0.1,y-0.5,typ, 'Parent',axh, 'FontName',axf)
end
%
drawnow()
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%bmPlotFig
function [idx,itp] = bmIndex(N,num,typ)
% Ensure exactly the same colors as the online ColorBrewer colorschemes.
%
itp = N>num;
switch typ
	case 'Qualitative'
		itp = false;
		idx = 1+mod(0:N-1,num);
	case 'Diverging'
		switch N
			case 1 % extrapolated
				idx = 8;
			case 2 % extrapolated
				idx = [4,12];
			case 3
				idx = [5,8,11];
			case 4
				idx = [3,6,10,13];
			case 5
				idx = [3,6,8,10,13];
			case 6
				idx = [2,5,7,9,11,14];
			case 7
				idx = [2,5,7,8,9,11,14];
			case 8
				idx = [2,4,6,7,9,10,12,14];
			case 9
				idx = [2,4,6,7,8,9,10,12,14];
			case 10
				idx = [1,2,4,6,7,9,10,12,14,15];
			otherwise
				idx = [1,2,4,6,7,8,9,10,12,14,15];
		end
	case 'Sequential'
		switch N
			case 1 % extrapolated
				idx = 6;
			case 2 % extrapolated
				idx = [4,8];
			case 3
				idx = [3,6,9];
			case 4
				idx = [2,5,7,10];
			case 5
				idx = [2,5,7,9,11];
			case 6
				idx = [2,4,6,7,9,11];
			case 7
				idx = [2,4,6,7,8,10,12];
			case 8
				idx = [1,3,4,6,7,8,10,12];
			otherwise
				idx = [1,3,4,6,7,8,10,11,13];
		end
	otherwise
		error('The colorscheme type "%s" is not recognized',typ)
end
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%bmIndex
function raw = bmColors()
% Return a structure of all colorschemes: name, scheme type, RGB values, number of nodes.
% Order: first sort by <typ>, then case-insensitive sort by <str>:
raw(35).str = 'YlOrRd';
raw(35).typ = 'Sequential';
raw(35).rgb = [255,255,204;255,255,178;255,237,160;254,217,118;254,204,92;254,178,76;253,141,60;252,78,42;240,59,32;227,26,28;189,0,38;177,0,38;128,0,38];
raw(34).str = 'YlOrBr';
raw(34).typ = 'Sequential';
raw(34).rgb = [255,255,229;255,255,212;255,247,188;254,227,145;254,217,142;254,196,79;254,153,41;236,112,20;217,95,14;204,76,2;153,52,4;140,45,4;102,37,6];
raw(33).str = 'YlGnBu';
raw(33).typ = 'Sequential';
raw(33).rgb = [255,255,217;255,255,204;237,248,177;199,233,180;161,218,180;127,205,187;65,182,196;29,145,192;44,127,184;34,94,168;37,52,148;12,44,132;8,29,88];
raw(32).str = 'YlGn';
raw(32).typ = 'Sequential';
raw(32).rgb = [255,255,229;255,255,204;247,252,185;217,240,163;194,230,153;173,221,142;120,198,121;65,171,93;49,163,84;35,132,67;0,104,55;0,90,50;0,69,41];
raw(31).str = 'Reds';
raw(31).typ = 'Sequential';
raw(31).rgb = [255,245,240;254,229,217;254,224,210;252,187,161;252,174,145;252,146,114;251,106,74;239,59,44;222,45,38;203,24,29;165,15,21;153,0,13;103,0,13];
raw(30).str = 'RdPu';
raw(30).typ = 'Sequential';
raw(30).rgb = [255,247,243;254,235,226;253,224,221;252,197,192;251,180,185;250,159,181;247,104,161;221,52,151;197,27,138;174,1,126;122,1,119;122,1,119;73,0,106];
raw(29).str = 'Purples';
raw(29).typ = 'Sequential';
raw(29).rgb = [252,251,253;242,240,247;239,237,245;218,218,235;203,201,226;188,189,220;158,154,200;128,125,186;117,107,177;106,81,163;84,39,143;74,20,134;63,0,125];
raw(28).str = 'PuRd';
raw(28).typ = 'Sequential';
raw(28).rgb = [247,244,249;241,238,246;231,225,239;212,185,218;215,181,216;201,148,199;223,101,176;231,41,138;221,28,119;206,18,86;152,0,67;145,0,63;103,0,31];
raw(27).str = 'PuBuGn';
raw(27).typ = 'Sequential';
raw(27).rgb = [255,247,251;246,239,247;236,226,240;208,209,230;189,201,225;166,189,219;103,169,207;54,144,192;28,144,153;2,129,138;1,108,89;1,100,80;1,70,54];
raw(26).str = 'PuBu';
raw(26).typ = 'Sequential';
raw(26).rgb = [255,247,251;241,238,246;236,231,242;208,209,230;189,201,225;166,189,219;116,169,207;54,144,192;43,140,190;5,112,176;4,90,141;3,78,123;2,56,88];
raw(25).str = 'Oranges';
raw(25).typ = 'Sequential';
raw(25).rgb = [255,245,235;254,237,222;254,230,206;253,208,162;253,190,133;253,174,107;253,141,60;241,105,19;230,85,13;217,72,1;166,54,3;140,45,4;127,39,4];
raw(24).str = 'OrRd';
raw(24).typ = 'Sequential';
raw(24).rgb = [255,247,236;254,240,217;254,232,200;253,212,158;253,204,138;253,187,132;252,141,89;239,101,72;227,74,51;215,48,31;179,0,0;153,0,0;127,0,0];
raw(23).str = 'Greys';
raw(23).typ = 'Sequential';
raw(23).rgb = [255,255,255;247,247,247;240,240,240;217,217,217;204,204,204;189,189,189;150,150,150;115,115,115;99,99,99;82,82,82;37,37,37;37,37,37;0,0,0];
raw(22).str = 'Greens';
raw(22).typ = 'Sequential';
raw(22).rgb = [247,252,245;237,248,233;229,245,224;199,233,192;186,228,179;161,217,155;116,196,118;65,171,93;49,163,84;35,139,69;0,109,44;0,90,50;0,68,27];
raw(21).str = 'GnBu';
raw(21).typ = 'Sequential';
raw(21).rgb = [247,252,240;240,249,232;224,243,219;204,235,197;186,228,188;168,221,181;123,204,196;78,179,211;67,162,202;43,140,190;8,104,172;8,88,158;8,64,129];
raw(20).str = 'BuPu';
raw(20).typ = 'Sequential';
raw(20).rgb = [247,252,253;237,248,251;224,236,244;191,211,230;179,205,227;158,188,218;140,150,198;140,107,177;136,86,167;136,65,157;129,15,124;110,1,107;77,0,75];
raw(19).str = 'BuGn';
raw(19).typ = 'Sequential';
raw(19).rgb = [247,252,253;237,248,251;229,245,249;204,236,230;178,226,226;153,216,201;102,194,164;65,174,118;44,162,95;35,139,69;0,109,44;0,88,36;0,68,27];
raw(18).str = 'Blues';
raw(18).typ = 'Sequential';
raw(18).rgb = [247,251,255;239,243,255;222,235,247;198,219,239;189,215,231;158,202,225;107,174,214;66,146,198;49,130,189;33,113,181;8,81,156;8,69,148;8,48,107];
raw(17).str = 'Set3';
raw(17).typ = 'Qualitative';
raw(17).rgb = [141,211,199;255,255,179;190,186,218;251,128,114;128,177,211;253,180,98;179,222,105;252,205,229;217,217,217;188,128,189;204,235,197;255,237,111];
raw(16).str = 'Set2';
raw(16).typ = 'Qualitative';
raw(16).rgb = [102,194,165;252,141,98;141,160,203;231,138,195;166,216,84;255,217,47;229,196,148;179,179,179];
raw(15).str = 'Set1';
raw(15).typ = 'Qualitative';
raw(15).rgb = [228,26,28;55,126,184;77,175,74;152,78,163;255,127,0;255,255,51;166,86,40;247,129,191;153,153,153];
raw(14).str = 'Pastel2';
raw(14).typ = 'Qualitative';
raw(14).rgb = [179,226,205;253,205,172;203,213,232;244,202,228;230,245,201;255,242,174;241,226,204;204,204,204];
raw(13).str = 'Pastel1';
raw(13).typ = 'Qualitative';
raw(13).rgb = [251,180,174;179,205,227;204,235,197;222,203,228;254,217,166;255,255,204;229,216,189;253,218,236;242,242,242];
raw(12).str = 'Paired';
raw(12).typ = 'Qualitative';
raw(12).rgb = [166,206,227;31,120,180;178,223,138;51,160,44;251,154,153;227,26,28;253,191,111;255,127,0;202,178,214;106,61,154;255,255,153;177,89,40];
raw(11).str = 'Dark2';
raw(11).typ = 'Qualitative';
raw(11).rgb = [27,158,119;217,95,2;117,112,179;231,41,138;102,166,30;230,171,2;166,118,29;102,102,102];
raw(10).str = 'Accent';
raw(10).typ = 'Qualitative';
raw(10).rgb = [127,201,127;190,174,212;253,192,134;255,255,153;56,108,176;240,2,127;191,91,23;102,102,102];
raw(09).str = 'Spectral';
raw(09).typ = 'Diverging';
raw(09).rgb = [158,1,66;213,62,79;215,25,28;244,109,67;252,141,89;253,174,97;254,224,139;255,255,191;230,245,152;171,221,164;153,213,148;102,194,165;43,131,186;50,136,189;94,79,162];
raw(08).str = 'RdYlGn';
raw(08).typ = 'Diverging';
raw(08).rgb = [165,0,38;215,48,39;215,25,28;244,109,67;252,141,89;253,174,97;254,224,139;255,255,191;217,239,139;166,217,106;145,207,96;102,189,99;26,150,65;26,152,80;0,104,55];
raw(07).str = 'RdYlBu';
raw(07).typ = 'Diverging';
raw(07).rgb = [165,0,38;215,48,39;215,25,28;244,109,67;252,141,89;253,174,97;254,224,144;255,255,191;224,243,248;171,217,233;145,191,219;116,173,209;44,123,182;69,117,180;49,54,149];
raw(06).str = 'RdGy';
raw(06).typ = 'Diverging';
raw(06).rgb = [103,0,31;178,24,43;202,0,32;214,96,77;239,138,98;244,165,130;253,219,199;255,255,255;224,224,224;186,186,186;153,153,153;135,135,135;64,64,64;77,77,77;26,26,26];
raw(05).str = 'RdBu';
raw(05).typ = 'Diverging';
raw(05).rgb = [103,0,31;178,24,43;202,0,32;214,96,77;239,138,98;244,165,130;253,219,199;247,247,247;209,229,240;146,197,222;103,169,207;67,147,195;5,113,176;33,102,172;5,48,97];
raw(04).str = 'PuOr';
raw(04).typ = 'Diverging';
raw(04).rgb = [127,59,8;179,88,6;230,97,1;224,130,20;241,163,64;253,184,99;254,224,182;247,247,247;216,218,235;178,171,210;153,142,195;128,115,172;94,60,153;84,39,136;45,0,75];
raw(03).str = 'PRGn';
raw(03).typ = 'Diverging';
raw(03).rgb = [64,0,75;118,42,131;123,50,148;153,112,171;175,141,195;194,165,207;231,212,232;247,247,247;217,240,211;166,219,160;127,191,123;90,174,97;0,136,55;27,120,55;0,68,27];
raw(02).str = 'PiYG';
raw(02).typ = 'Diverging';
raw(02).rgb = [142,1,82;197,27,125;208,28,139;222,119,174;233,163,201;241,182,218;253,224,239;247,247,247;230,245,208;184,225,134;161,215,106;127,188,65;77,172,38;77,146,33;39,100,25];
raw(01).str = 'BrBG';
raw(01).typ = 'Diverging';
raw(01).rgb = [84,48,5;140,81,10;166,97,26;191,129,45;216,179,101;223,194,125;246,232,195;245,245,245;199,234,229;128,205,193;90,180,172;53,151,143;1,133,113;1,102,94;0,60,48];




% number of nodes:
for k = 1:numel(raw)
	switch raw(k).typ
		case 'Diverging'
			raw(k).num = 11;
		case 'Qualitative'
			raw(k).num = size(raw(k).rgb,1);
		case 'Sequential'
			raw(k).num = 9;
	end
end
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%bmColors
% Code and Implementation:
% Copyright (c) 2014 Stephen Cobeldick
% Color Values Only:
% Copyright (c) 2002 Cynthia Brewer, Mark Harrower, and The Pennsylvania State University.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and limitations under the License.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions as source code must retain the above copyright notice, this
% list of conditions and the following disclaimer.
%
% 2. The end-user documentation included with the redistribution, if any, must
% include the following acknowledgment: "This product includes color
% specifications and designs developed by Cynthia Brewer
% (http://colorbrewer.org/)." Alternately, this acknowledgment may appear in the
% software itself, if and wherever such third-party acknowledgments normally appear.
%
% 4. The name "ColorBrewer" must not be used to endorse or promote products
% derived from this software without prior written permission. For written
% permission, please contact Cynthia Brewer at cbrewer@psu.edu.
%
% 5. Products derived from this software may not be called "ColorBrewer", nor
% may "ColorBrewer" appear in their name, without prior written permission of Cynthia Brewer.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%license

function cm_data=viridis(m)
cm = [[ 0.26700401,  0.00487433,  0.32941519],
       [ 0.26851048,  0.00960483,  0.33542652],
       [ 0.26994384,  0.01462494,  0.34137895],
       [ 0.27130489,  0.01994186,  0.34726862],
       [ 0.27259384,  0.02556309,  0.35309303],
       [ 0.27380934,  0.03149748,  0.35885256],
       [ 0.27495242,  0.03775181,  0.36454323],
       [ 0.27602238,  0.04416723,  0.37016418],
       [ 0.2770184 ,  0.05034437,  0.37571452],
       [ 0.27794143,  0.05632444,  0.38119074],
       [ 0.27879067,  0.06214536,  0.38659204],
       [ 0.2795655 ,  0.06783587,  0.39191723],
       [ 0.28026658,  0.07341724,  0.39716349],
       [ 0.28089358,  0.07890703,  0.40232944],
       [ 0.28144581,  0.0843197 ,  0.40741404],
       [ 0.28192358,  0.08966622,  0.41241521],
       [ 0.28232739,  0.09495545,  0.41733086],
       [ 0.28265633,  0.10019576,  0.42216032],
       [ 0.28291049,  0.10539345,  0.42690202],
       [ 0.28309095,  0.11055307,  0.43155375],
       [ 0.28319704,  0.11567966,  0.43611482],
       [ 0.28322882,  0.12077701,  0.44058404],
       [ 0.28318684,  0.12584799,  0.44496   ],
       [ 0.283072  ,  0.13089477,  0.44924127],
       [ 0.28288389,  0.13592005,  0.45342734],
       [ 0.28262297,  0.14092556,  0.45751726],
       [ 0.28229037,  0.14591233,  0.46150995],
       [ 0.28188676,  0.15088147,  0.46540474],
       [ 0.28141228,  0.15583425,  0.46920128],
       [ 0.28086773,  0.16077132,  0.47289909],
       [ 0.28025468,  0.16569272,  0.47649762],
       [ 0.27957399,  0.17059884,  0.47999675],
       [ 0.27882618,  0.1754902 ,  0.48339654],
       [ 0.27801236,  0.18036684,  0.48669702],
       [ 0.27713437,  0.18522836,  0.48989831],
       [ 0.27619376,  0.19007447,  0.49300074],
       [ 0.27519116,  0.1949054 ,  0.49600488],
       [ 0.27412802,  0.19972086,  0.49891131],
       [ 0.27300596,  0.20452049,  0.50172076],
       [ 0.27182812,  0.20930306,  0.50443413],
       [ 0.27059473,  0.21406899,  0.50705243],
       [ 0.26930756,  0.21881782,  0.50957678],
       [ 0.26796846,  0.22354911,  0.5120084 ],
       [ 0.26657984,  0.2282621 ,  0.5143487 ],
       [ 0.2651445 ,  0.23295593,  0.5165993 ],
       [ 0.2636632 ,  0.23763078,  0.51876163],
       [ 0.26213801,  0.24228619,  0.52083736],
       [ 0.26057103,  0.2469217 ,  0.52282822],
       [ 0.25896451,  0.25153685,  0.52473609],
       [ 0.25732244,  0.2561304 ,  0.52656332],
       [ 0.25564519,  0.26070284,  0.52831152],
       [ 0.25393498,  0.26525384,  0.52998273],
       [ 0.25219404,  0.26978306,  0.53157905],
       [ 0.25042462,  0.27429024,  0.53310261],
       [ 0.24862899,  0.27877509,  0.53455561],
       [ 0.2468114 ,  0.28323662,  0.53594093],
       [ 0.24497208,  0.28767547,  0.53726018],
       [ 0.24311324,  0.29209154,  0.53851561],
       [ 0.24123708,  0.29648471,  0.53970946],
       [ 0.23934575,  0.30085494,  0.54084398],
       [ 0.23744138,  0.30520222,  0.5419214 ],
       [ 0.23552606,  0.30952657,  0.54294396],
       [ 0.23360277,  0.31382773,  0.54391424],
       [ 0.2316735 ,  0.3181058 ,  0.54483444],
       [ 0.22973926,  0.32236127,  0.54570633],
       [ 0.22780192,  0.32659432,  0.546532  ],
       [ 0.2258633 ,  0.33080515,  0.54731353],
       [ 0.22392515,  0.334994  ,  0.54805291],
       [ 0.22198915,  0.33916114,  0.54875211],
       [ 0.22005691,  0.34330688,  0.54941304],
       [ 0.21812995,  0.34743154,  0.55003755],
       [ 0.21620971,  0.35153548,  0.55062743],
       [ 0.21429757,  0.35561907,  0.5511844 ],
       [ 0.21239477,  0.35968273,  0.55171011],
       [ 0.2105031 ,  0.36372671,  0.55220646],
       [ 0.20862342,  0.36775151,  0.55267486],
       [ 0.20675628,  0.37175775,  0.55311653],
       [ 0.20490257,  0.37574589,  0.55353282],
       [ 0.20306309,  0.37971644,  0.55392505],
       [ 0.20123854,  0.38366989,  0.55429441],
       [ 0.1994295 ,  0.38760678,  0.55464205],
       [ 0.1976365 ,  0.39152762,  0.55496905],
       [ 0.19585993,  0.39543297,  0.55527637],
       [ 0.19410009,  0.39932336,  0.55556494],
       [ 0.19235719,  0.40319934,  0.55583559],
       [ 0.19063135,  0.40706148,  0.55608907],
       [ 0.18892259,  0.41091033,  0.55632606],
       [ 0.18723083,  0.41474645,  0.55654717],
       [ 0.18555593,  0.4185704 ,  0.55675292],
       [ 0.18389763,  0.42238275,  0.55694377],
       [ 0.18225561,  0.42618405,  0.5571201 ],
       [ 0.18062949,  0.42997486,  0.55728221],
       [ 0.17901879,  0.43375572,  0.55743035],
       [ 0.17742298,  0.4375272 ,  0.55756466],
       [ 0.17584148,  0.44128981,  0.55768526],
       [ 0.17427363,  0.4450441 ,  0.55779216],
       [ 0.17271876,  0.4487906 ,  0.55788532],
       [ 0.17117615,  0.4525298 ,  0.55796464],
       [ 0.16964573,  0.45626209,  0.55803034],
       [ 0.16812641,  0.45998802,  0.55808199],
       [ 0.1666171 ,  0.46370813,  0.55811913],
       [ 0.16511703,  0.4674229 ,  0.55814141],
       [ 0.16362543,  0.47113278,  0.55814842],
       [ 0.16214155,  0.47483821,  0.55813967],
       [ 0.16066467,  0.47853961,  0.55811466],
       [ 0.15919413,  0.4822374 ,  0.5580728 ],
       [ 0.15772933,  0.48593197,  0.55801347],
       [ 0.15626973,  0.4896237 ,  0.557936  ],
       [ 0.15481488,  0.49331293,  0.55783967],
       [ 0.15336445,  0.49700003,  0.55772371],
       [ 0.1519182 ,  0.50068529,  0.55758733],
       [ 0.15047605,  0.50436904,  0.55742968],
       [ 0.14903918,  0.50805136,  0.5572505 ],
       [ 0.14760731,  0.51173263,  0.55704861],
       [ 0.14618026,  0.51541316,  0.55682271],
       [ 0.14475863,  0.51909319,  0.55657181],
       [ 0.14334327,  0.52277292,  0.55629491],
       [ 0.14193527,  0.52645254,  0.55599097],
       [ 0.14053599,  0.53013219,  0.55565893],
       [ 0.13914708,  0.53381201,  0.55529773],
       [ 0.13777048,  0.53749213,  0.55490625],
       [ 0.1364085 ,  0.54117264,  0.55448339],
       [ 0.13506561,  0.54485335,  0.55402906],
       [ 0.13374299,  0.54853458,  0.55354108],
       [ 0.13244401,  0.55221637,  0.55301828],
       [ 0.13117249,  0.55589872,  0.55245948],
       [ 0.1299327 ,  0.55958162,  0.55186354],
       [ 0.12872938,  0.56326503,  0.55122927],
       [ 0.12756771,  0.56694891,  0.55055551],
       [ 0.12645338,  0.57063316,  0.5498411 ],
       [ 0.12539383,  0.57431754,  0.54908564],
       [ 0.12439474,  0.57800205,  0.5482874 ],
       [ 0.12346281,  0.58168661,  0.54744498],
       [ 0.12260562,  0.58537105,  0.54655722],
       [ 0.12183122,  0.58905521,  0.54562298],
       [ 0.12114807,  0.59273889,  0.54464114],
       [ 0.12056501,  0.59642187,  0.54361058],
       [ 0.12009154,  0.60010387,  0.54253043],
       [ 0.11973756,  0.60378459,  0.54139999],
       [ 0.11951163,  0.60746388,  0.54021751],
       [ 0.11942341,  0.61114146,  0.53898192],
       [ 0.11948255,  0.61481702,  0.53769219],
       [ 0.11969858,  0.61849025,  0.53634733],
       [ 0.12008079,  0.62216081,  0.53494633],
       [ 0.12063824,  0.62582833,  0.53348834],
       [ 0.12137972,  0.62949242,  0.53197275],
       [ 0.12231244,  0.63315277,  0.53039808],
       [ 0.12344358,  0.63680899,  0.52876343],
       [ 0.12477953,  0.64046069,  0.52706792],
       [ 0.12632581,  0.64410744,  0.52531069],
       [ 0.12808703,  0.64774881,  0.52349092],
       [ 0.13006688,  0.65138436,  0.52160791],
       [ 0.13226797,  0.65501363,  0.51966086],
       [ 0.13469183,  0.65863619,  0.5176488 ],
       [ 0.13733921,  0.66225157,  0.51557101],
       [ 0.14020991,  0.66585927,  0.5134268 ],
       [ 0.14330291,  0.66945881,  0.51121549],
       [ 0.1466164 ,  0.67304968,  0.50893644],
       [ 0.15014782,  0.67663139,  0.5065889 ],
       [ 0.15389405,  0.68020343,  0.50417217],
       [ 0.15785146,  0.68376525,  0.50168574],
       [ 0.16201598,  0.68731632,  0.49912906],
       [ 0.1663832 ,  0.69085611,  0.49650163],
       [ 0.1709484 ,  0.69438405,  0.49380294],
       [ 0.17570671,  0.6978996 ,  0.49103252],
       [ 0.18065314,  0.70140222,  0.48818938],
       [ 0.18578266,  0.70489133,  0.48527326],
       [ 0.19109018,  0.70836635,  0.48228395],
       [ 0.19657063,  0.71182668,  0.47922108],
       [ 0.20221902,  0.71527175,  0.47608431],
       [ 0.20803045,  0.71870095,  0.4728733 ],
       [ 0.21400015,  0.72211371,  0.46958774],
       [ 0.22012381,  0.72550945,  0.46622638],
       [ 0.2263969 ,  0.72888753,  0.46278934],
       [ 0.23281498,  0.73224735,  0.45927675],
       [ 0.2393739 ,  0.73558828,  0.45568838],
       [ 0.24606968,  0.73890972,  0.45202405],
       [ 0.25289851,  0.74221104,  0.44828355],
       [ 0.25985676,  0.74549162,  0.44446673],
       [ 0.26694127,  0.74875084,  0.44057284],
       [ 0.27414922,  0.75198807,  0.4366009 ],
       [ 0.28147681,  0.75520266,  0.43255207],
       [ 0.28892102,  0.75839399,  0.42842626],
       [ 0.29647899,  0.76156142,  0.42422341],
       [ 0.30414796,  0.76470433,  0.41994346],
       [ 0.31192534,  0.76782207,  0.41558638],
       [ 0.3198086 ,  0.77091403,  0.41115215],
       [ 0.3277958 ,  0.77397953,  0.40664011],
       [ 0.33588539,  0.7770179 ,  0.40204917],
       [ 0.34407411,  0.78002855,  0.39738103],
       [ 0.35235985,  0.78301086,  0.39263579],
       [ 0.36074053,  0.78596419,  0.38781353],
       [ 0.3692142 ,  0.78888793,  0.38291438],
       [ 0.37777892,  0.79178146,  0.3779385 ],
       [ 0.38643282,  0.79464415,  0.37288606],
       [ 0.39517408,  0.79747541,  0.36775726],
       [ 0.40400101,  0.80027461,  0.36255223],
       [ 0.4129135 ,  0.80304099,  0.35726893],
       [ 0.42190813,  0.80577412,  0.35191009],
       [ 0.43098317,  0.80847343,  0.34647607],
       [ 0.44013691,  0.81113836,  0.3409673 ],
       [ 0.44936763,  0.81376835,  0.33538426],
       [ 0.45867362,  0.81636288,  0.32972749],
       [ 0.46805314,  0.81892143,  0.32399761],
       [ 0.47750446,  0.82144351,  0.31819529],
       [ 0.4870258 ,  0.82392862,  0.31232133],
       [ 0.49661536,  0.82637633,  0.30637661],
       [ 0.5062713 ,  0.82878621,  0.30036211],
       [ 0.51599182,  0.83115784,  0.29427888],
       [ 0.52577622,  0.83349064,  0.2881265 ],
       [ 0.5356211 ,  0.83578452,  0.28190832],
       [ 0.5455244 ,  0.83803918,  0.27562602],
       [ 0.55548397,  0.84025437,  0.26928147],
       [ 0.5654976 ,  0.8424299 ,  0.26287683],
       [ 0.57556297,  0.84456561,  0.25641457],
       [ 0.58567772,  0.84666139,  0.24989748],
       [ 0.59583934,  0.84871722,  0.24332878],
       [ 0.60604528,  0.8507331 ,  0.23671214],
       [ 0.61629283,  0.85270912,  0.23005179],
       [ 0.62657923,  0.85464543,  0.22335258],
       [ 0.63690157,  0.85654226,  0.21662012],
       [ 0.64725685,  0.85839991,  0.20986086],
       [ 0.65764197,  0.86021878,  0.20308229],
       [ 0.66805369,  0.86199932,  0.19629307],
       [ 0.67848868,  0.86374211,  0.18950326],
       [ 0.68894351,  0.86544779,  0.18272455],
       [ 0.69941463,  0.86711711,  0.17597055],
       [ 0.70989842,  0.86875092,  0.16925712],
       [ 0.72039115,  0.87035015,  0.16260273],
       [ 0.73088902,  0.87191584,  0.15602894],
       [ 0.74138803,  0.87344918,  0.14956101],
       [ 0.75188414,  0.87495143,  0.14322828],
       [ 0.76237342,  0.87642392,  0.13706449],
       [ 0.77285183,  0.87786808,  0.13110864],
       [ 0.78331535,  0.87928545,  0.12540538],
       [ 0.79375994,  0.88067763,  0.12000532],
       [ 0.80418159,  0.88204632,  0.11496505],
       [ 0.81457634,  0.88339329,  0.11034678],
       [ 0.82494028,  0.88472036,  0.10621724],
       [ 0.83526959,  0.88602943,  0.1026459 ],
       [ 0.84556056,  0.88732243,  0.09970219],
       [ 0.8558096 ,  0.88860134,  0.09745186],
       [ 0.86601325,  0.88986815,  0.09595277],
       [ 0.87616824,  0.89112487,  0.09525046],
       [ 0.88627146,  0.89237353,  0.09537439],
       [ 0.89632002,  0.89361614,  0.09633538],
       [ 0.90631121,  0.89485467,  0.09812496],
       [ 0.91624212,  0.89609127,  0.1007168 ],
       [ 0.92610579,  0.89732977,  0.10407067],
       [ 0.93590444,  0.8985704 ,  0.10813094],
       [ 0.94563626,  0.899815  ,  0.11283773],
       [ 0.95529972,  0.90106534,  0.11812832],
       [ 0.96489353,  0.90232311,  0.12394051],
       [ 0.97441665,  0.90358991,  0.13021494],
       [ 0.98386829,  0.90486726,  0.13689671],
       [ 0.99324789,  0.90615657,  0.1439362 ]];

if nargin < 1
    cm_data = cm;
else
    hsv=rgb2hsv(cm);
    cm_data=interp1(linspace(0,1,size(cm,1)),hsv,linspace(0,1,m));
    cm_data=hsv2rgb(cm_data);
end
end


function cm_data=plasma(m)

cm = [[  5.03832136e-02,   2.98028976e-02,   5.27974883e-01],
       [  6.35363639e-02,   2.84259729e-02,   5.33123681e-01],
       [  7.53531234e-02,   2.72063728e-02,   5.38007001e-01],
       [  8.62217979e-02,   2.61253206e-02,   5.42657691e-01],
       [  9.63786097e-02,   2.51650976e-02,   5.47103487e-01],
       [  1.05979704e-01,   2.43092436e-02,   5.51367851e-01],
       [  1.15123641e-01,   2.35562500e-02,   5.55467728e-01],
       [  1.23902903e-01,   2.28781011e-02,   5.59423480e-01],
       [  1.32380720e-01,   2.22583774e-02,   5.63250116e-01],
       [  1.40603076e-01,   2.16866674e-02,   5.66959485e-01],
       [  1.48606527e-01,   2.11535876e-02,   5.70561711e-01],
       [  1.56420649e-01,   2.06507174e-02,   5.74065446e-01],
       [  1.64069722e-01,   2.01705326e-02,   5.77478074e-01],
       [  1.71573925e-01,   1.97063415e-02,   5.80805890e-01],
       [  1.78950212e-01,   1.92522243e-02,   5.84054243e-01],
       [  1.86212958e-01,   1.88029767e-02,   5.87227661e-01],
       [  1.93374449e-01,   1.83540593e-02,   5.90329954e-01],
       [  2.00445260e-01,   1.79015512e-02,   5.93364304e-01],
       [  2.07434551e-01,   1.74421086e-02,   5.96333341e-01],
       [  2.14350298e-01,   1.69729276e-02,   5.99239207e-01],
       [  2.21196750e-01,   1.64970484e-02,   6.02083323e-01],
       [  2.27982971e-01,   1.60071509e-02,   6.04867403e-01],
       [  2.34714537e-01,   1.55015065e-02,   6.07592438e-01],
       [  2.41396253e-01,   1.49791041e-02,   6.10259089e-01],
       [  2.48032377e-01,   1.44393586e-02,   6.12867743e-01],
       [  2.54626690e-01,   1.38820918e-02,   6.15418537e-01],
       [  2.61182562e-01,   1.33075156e-02,   6.17911385e-01],
       [  2.67702993e-01,   1.27162163e-02,   6.20345997e-01],
       [  2.74190665e-01,   1.21091423e-02,   6.22721903e-01],
       [  2.80647969e-01,   1.14875915e-02,   6.25038468e-01],
       [  2.87076059e-01,   1.08554862e-02,   6.27294975e-01],
       [  2.93477695e-01,   1.02128849e-02,   6.29490490e-01],
       [  2.99855122e-01,   9.56079551e-03,   6.31623923e-01],
       [  3.06209825e-01,   8.90185346e-03,   6.33694102e-01],
       [  3.12543124e-01,   8.23900704e-03,   6.35699759e-01],
       [  3.18856183e-01,   7.57551051e-03,   6.37639537e-01],
       [  3.25150025e-01,   6.91491734e-03,   6.39512001e-01],
       [  3.31425547e-01,   6.26107379e-03,   6.41315649e-01],
       [  3.37683446e-01,   5.61830889e-03,   6.43048936e-01],
       [  3.43924591e-01,   4.99053080e-03,   6.44710195e-01],
       [  3.50149699e-01,   4.38202557e-03,   6.46297711e-01],
       [  3.56359209e-01,   3.79781761e-03,   6.47809772e-01],
       [  3.62553473e-01,   3.24319591e-03,   6.49244641e-01],
       [  3.68732762e-01,   2.72370721e-03,   6.50600561e-01],
       [  3.74897270e-01,   2.24514897e-03,   6.51875762e-01],
       [  3.81047116e-01,   1.81356205e-03,   6.53068467e-01],
       [  3.87182639e-01,   1.43446923e-03,   6.54176761e-01],
       [  3.93304010e-01,   1.11388259e-03,   6.55198755e-01],
       [  3.99410821e-01,   8.59420809e-04,   6.56132835e-01],
       [  4.05502914e-01,   6.78091517e-04,   6.56977276e-01],
       [  4.11580082e-01,   5.77101735e-04,   6.57730380e-01],
       [  4.17642063e-01,   5.63847476e-04,   6.58390492e-01],
       [  4.23688549e-01,   6.45902780e-04,   6.58956004e-01],
       [  4.29719186e-01,   8.31008207e-04,   6.59425363e-01],
       [  4.35733575e-01,   1.12705875e-03,   6.59797077e-01],
       [  4.41732123e-01,   1.53984779e-03,   6.60069009e-01],
       [  4.47713600e-01,   2.07954744e-03,   6.60240367e-01],
       [  4.53677394e-01,   2.75470302e-03,   6.60309966e-01],
       [  4.59622938e-01,   3.57374415e-03,   6.60276655e-01],
       [  4.65549631e-01,   4.54518084e-03,   6.60139383e-01],
       [  4.71456847e-01,   5.67758762e-03,   6.59897210e-01],
       [  4.77343929e-01,   6.97958743e-03,   6.59549311e-01],
       [  4.83210198e-01,   8.45983494e-03,   6.59094989e-01],
       [  4.89054951e-01,   1.01269996e-02,   6.58533677e-01],
       [  4.94877466e-01,   1.19897486e-02,   6.57864946e-01],
       [  5.00677687e-01,   1.40550640e-02,   6.57087561e-01],
       [  5.06454143e-01,   1.63333443e-02,   6.56202294e-01],
       [  5.12206035e-01,   1.88332232e-02,   6.55209222e-01],
       [  5.17932580e-01,   2.15631918e-02,   6.54108545e-01],
       [  5.23632990e-01,   2.45316468e-02,   6.52900629e-01],
       [  5.29306474e-01,   2.77468735e-02,   6.51586010e-01],
       [  5.34952244e-01,   3.12170300e-02,   6.50165396e-01],
       [  5.40569510e-01,   3.49501310e-02,   6.48639668e-01],
       [  5.46157494e-01,   3.89540334e-02,   6.47009884e-01],
       [  5.51715423e-01,   4.31364795e-02,   6.45277275e-01],
       [  5.57242538e-01,   4.73307585e-02,   6.43443250e-01],
       [  5.62738096e-01,   5.15448092e-02,   6.41509389e-01],
       [  5.68201372e-01,   5.57776706e-02,   6.39477440e-01],
       [  5.73631859e-01,   6.00281369e-02,   6.37348841e-01],
       [  5.79028682e-01,   6.42955547e-02,   6.35126108e-01],
       [  5.84391137e-01,   6.85790261e-02,   6.32811608e-01],
       [  5.89718606e-01,   7.28775875e-02,   6.30407727e-01],
       [  5.95010505e-01,   7.71902878e-02,   6.27916992e-01],
       [  6.00266283e-01,   8.15161895e-02,   6.25342058e-01],
       [  6.05485428e-01,   8.58543713e-02,   6.22685703e-01],
       [  6.10667469e-01,   9.02039303e-02,   6.19950811e-01],
       [  6.15811974e-01,   9.45639838e-02,   6.17140367e-01],
       [  6.20918555e-01,   9.89336721e-02,   6.14257440e-01],
       [  6.25986869e-01,   1.03312160e-01,   6.11305174e-01],
       [  6.31016615e-01,   1.07698641e-01,   6.08286774e-01],
       [  6.36007543e-01,   1.12092335e-01,   6.05205491e-01],
       [  6.40959444e-01,   1.16492495e-01,   6.02064611e-01],
       [  6.45872158e-01,   1.20898405e-01,   5.98867442e-01],
       [  6.50745571e-01,   1.25309384e-01,   5.95617300e-01],
       [  6.55579615e-01,   1.29724785e-01,   5.92317494e-01],
       [  6.60374266e-01,   1.34143997e-01,   5.88971318e-01],
       [  6.65129493e-01,   1.38566428e-01,   5.85582301e-01],
       [  6.69845385e-01,   1.42991540e-01,   5.82153572e-01],
       [  6.74522060e-01,   1.47418835e-01,   5.78688247e-01],
       [  6.79159664e-01,   1.51847851e-01,   5.75189431e-01],
       [  6.83758384e-01,   1.56278163e-01,   5.71660158e-01],
       [  6.88318440e-01,   1.60709387e-01,   5.68103380e-01],
       [  6.92840088e-01,   1.65141174e-01,   5.64521958e-01],
       [  6.97323615e-01,   1.69573215e-01,   5.60918659e-01],
       [  7.01769334e-01,   1.74005236e-01,   5.57296144e-01],
       [  7.06177590e-01,   1.78437000e-01,   5.53656970e-01],
       [  7.10548747e-01,   1.82868306e-01,   5.50003579e-01],
       [  7.14883195e-01,   1.87298986e-01,   5.46338299e-01],
       [  7.19181339e-01,   1.91728906e-01,   5.42663338e-01],
       [  7.23443604e-01,   1.96157962e-01,   5.38980786e-01],
       [  7.27670428e-01,   2.00586086e-01,   5.35292612e-01],
       [  7.31862231e-01,   2.05013174e-01,   5.31600995e-01],
       [  7.36019424e-01,   2.09439071e-01,   5.27908434e-01],
       [  7.40142557e-01,   2.13863965e-01,   5.24215533e-01],
       [  7.44232102e-01,   2.18287899e-01,   5.20523766e-01],
       [  7.48288533e-01,   2.22710942e-01,   5.16834495e-01],
       [  7.52312321e-01,   2.27133187e-01,   5.13148963e-01],
       [  7.56303937e-01,   2.31554749e-01,   5.09468305e-01],
       [  7.60263849e-01,   2.35975765e-01,   5.05793543e-01],
       [  7.64192516e-01,   2.40396394e-01,   5.02125599e-01],
       [  7.68090391e-01,   2.44816813e-01,   4.98465290e-01],
       [  7.71957916e-01,   2.49237220e-01,   4.94813338e-01],
       [  7.75795522e-01,   2.53657797e-01,   4.91170517e-01],
       [  7.79603614e-01,   2.58078397e-01,   4.87539124e-01],
       [  7.83382636e-01,   2.62499662e-01,   4.83917732e-01],
       [  7.87132978e-01,   2.66921859e-01,   4.80306702e-01],
       [  7.90855015e-01,   2.71345267e-01,   4.76706319e-01],
       [  7.94549101e-01,   2.75770179e-01,   4.73116798e-01],
       [  7.98215577e-01,   2.80196901e-01,   4.69538286e-01],
       [  8.01854758e-01,   2.84625750e-01,   4.65970871e-01],
       [  8.05466945e-01,   2.89057057e-01,   4.62414580e-01],
       [  8.09052419e-01,   2.93491117e-01,   4.58869577e-01],
       [  8.12611506e-01,   2.97927865e-01,   4.55337565e-01],
       [  8.16144382e-01,   3.02368130e-01,   4.51816385e-01],
       [  8.19651255e-01,   3.06812282e-01,   4.48305861e-01],
       [  8.23132309e-01,   3.11260703e-01,   4.44805781e-01],
       [  8.26587706e-01,   3.15713782e-01,   4.41315901e-01],
       [  8.30017584e-01,   3.20171913e-01,   4.37835947e-01],
       [  8.33422053e-01,   3.24635499e-01,   4.34365616e-01],
       [  8.36801237e-01,   3.29104836e-01,   4.30905052e-01],
       [  8.40155276e-01,   3.33580106e-01,   4.27454836e-01],
       [  8.43484103e-01,   3.38062109e-01,   4.24013059e-01],
       [  8.46787726e-01,   3.42551272e-01,   4.20579333e-01],
       [  8.50066132e-01,   3.47048028e-01,   4.17153264e-01],
       [  8.53319279e-01,   3.51552815e-01,   4.13734445e-01],
       [  8.56547103e-01,   3.56066072e-01,   4.10322469e-01],
       [  8.59749520e-01,   3.60588229e-01,   4.06916975e-01],
       [  8.62926559e-01,   3.65119408e-01,   4.03518809e-01],
       [  8.66077920e-01,   3.69660446e-01,   4.00126027e-01],
       [  8.69203436e-01,   3.74211795e-01,   3.96738211e-01],
       [  8.72302917e-01,   3.78773910e-01,   3.93354947e-01],
       [  8.75376149e-01,   3.83347243e-01,   3.89975832e-01],
       [  8.78422895e-01,   3.87932249e-01,   3.86600468e-01],
       [  8.81442916e-01,   3.92529339e-01,   3.83228622e-01],
       [  8.84435982e-01,   3.97138877e-01,   3.79860246e-01],
       [  8.87401682e-01,   4.01761511e-01,   3.76494232e-01],
       [  8.90339687e-01,   4.06397694e-01,   3.73130228e-01],
       [  8.93249647e-01,   4.11047871e-01,   3.69767893e-01],
       [  8.96131191e-01,   4.15712489e-01,   3.66406907e-01],
       [  8.98983931e-01,   4.20391986e-01,   3.63046965e-01],
       [  9.01807455e-01,   4.25086807e-01,   3.59687758e-01],
       [  9.04601295e-01,   4.29797442e-01,   3.56328796e-01],
       [  9.07364995e-01,   4.34524335e-01,   3.52969777e-01],
       [  9.10098088e-01,   4.39267908e-01,   3.49610469e-01],
       [  9.12800095e-01,   4.44028574e-01,   3.46250656e-01],
       [  9.15470518e-01,   4.48806744e-01,   3.42890148e-01],
       [  9.18108848e-01,   4.53602818e-01,   3.39528771e-01],
       [  9.20714383e-01,   4.58417420e-01,   3.36165582e-01],
       [  9.23286660e-01,   4.63250828e-01,   3.32800827e-01],
       [  9.25825146e-01,   4.68103387e-01,   3.29434512e-01],
       [  9.28329275e-01,   4.72975465e-01,   3.26066550e-01],
       [  9.30798469e-01,   4.77867420e-01,   3.22696876e-01],
       [  9.33232140e-01,   4.82779603e-01,   3.19325444e-01],
       [  9.35629684e-01,   4.87712357e-01,   3.15952211e-01],
       [  9.37990034e-01,   4.92666544e-01,   3.12575440e-01],
       [  9.40312939e-01,   4.97642038e-01,   3.09196628e-01],
       [  9.42597771e-01,   5.02639147e-01,   3.05815824e-01],
       [  9.44843893e-01,   5.07658169e-01,   3.02433101e-01],
       [  9.47050662e-01,   5.12699390e-01,   2.99048555e-01],
       [  9.49217427e-01,   5.17763087e-01,   2.95662308e-01],
       [  9.51343530e-01,   5.22849522e-01,   2.92274506e-01],
       [  9.53427725e-01,   5.27959550e-01,   2.88883445e-01],
       [  9.55469640e-01,   5.33093083e-01,   2.85490391e-01],
       [  9.57468770e-01,   5.38250172e-01,   2.82096149e-01],
       [  9.59424430e-01,   5.43431038e-01,   2.78700990e-01],
       [  9.61335930e-01,   5.48635890e-01,   2.75305214e-01],
       [  9.63202573e-01,   5.53864931e-01,   2.71909159e-01],
       [  9.65023656e-01,   5.59118349e-01,   2.68513200e-01],
       [  9.66798470e-01,   5.64396327e-01,   2.65117752e-01],
       [  9.68525639e-01,   5.69699633e-01,   2.61721488e-01],
       [  9.70204593e-01,   5.75028270e-01,   2.58325424e-01],
       [  9.71835007e-01,   5.80382015e-01,   2.54931256e-01],
       [  9.73416145e-01,   5.85761012e-01,   2.51539615e-01],
       [  9.74947262e-01,   5.91165394e-01,   2.48151200e-01],
       [  9.76427606e-01,   5.96595287e-01,   2.44766775e-01],
       [  9.77856416e-01,   6.02050811e-01,   2.41387186e-01],
       [  9.79232922e-01,   6.07532077e-01,   2.38013359e-01],
       [  9.80556344e-01,   6.13039190e-01,   2.34646316e-01],
       [  9.81825890e-01,   6.18572250e-01,   2.31287178e-01],
       [  9.83040742e-01,   6.24131362e-01,   2.27937141e-01],
       [  9.84198924e-01,   6.29717516e-01,   2.24595006e-01],
       [  9.85300760e-01,   6.35329876e-01,   2.21264889e-01],
       [  9.86345421e-01,   6.40968508e-01,   2.17948456e-01],
       [  9.87332067e-01,   6.46633475e-01,   2.14647532e-01],
       [  9.88259846e-01,   6.52324832e-01,   2.11364122e-01],
       [  9.89127893e-01,   6.58042630e-01,   2.08100426e-01],
       [  9.89935328e-01,   6.63786914e-01,   2.04858855e-01],
       [  9.90681261e-01,   6.69557720e-01,   2.01642049e-01],
       [  9.91364787e-01,   6.75355082e-01,   1.98452900e-01],
       [  9.91984990e-01,   6.81179025e-01,   1.95294567e-01],
       [  9.92540939e-01,   6.87029567e-01,   1.92170500e-01],
       [  9.93031693e-01,   6.92906719e-01,   1.89084459e-01],
       [  9.93456302e-01,   6.98810484e-01,   1.86040537e-01],
       [  9.93813802e-01,   7.04740854e-01,   1.83043180e-01],
       [  9.94103226e-01,   7.10697814e-01,   1.80097207e-01],
       [  9.94323596e-01,   7.16681336e-01,   1.77207826e-01],
       [  9.94473934e-01,   7.22691379e-01,   1.74380656e-01],
       [  9.94553260e-01,   7.28727890e-01,   1.71621733e-01],
       [  9.94560594e-01,   7.34790799e-01,   1.68937522e-01],
       [  9.94494964e-01,   7.40880020e-01,   1.66334918e-01],
       [  9.94355411e-01,   7.46995448e-01,   1.63821243e-01],
       [  9.94140989e-01,   7.53136955e-01,   1.61404226e-01],
       [  9.93850778e-01,   7.59304390e-01,   1.59091984e-01],
       [  9.93482190e-01,   7.65498551e-01,   1.56890625e-01],
       [  9.93033251e-01,   7.71719833e-01,   1.54807583e-01],
       [  9.92505214e-01,   7.77966775e-01,   1.52854862e-01],
       [  9.91897270e-01,   7.84239120e-01,   1.51041581e-01],
       [  9.91208680e-01,   7.90536569e-01,   1.49376885e-01],
       [  9.90438793e-01,   7.96858775e-01,   1.47869810e-01],
       [  9.89587065e-01,   8.03205337e-01,   1.46529128e-01],
       [  9.88647741e-01,   8.09578605e-01,   1.45357284e-01],
       [  9.87620557e-01,   8.15977942e-01,   1.44362644e-01],
       [  9.86509366e-01,   8.22400620e-01,   1.43556679e-01],
       [  9.85314198e-01,   8.28845980e-01,   1.42945116e-01],
       [  9.84031139e-01,   8.35315360e-01,   1.42528388e-01],
       [  9.82652820e-01,   8.41811730e-01,   1.42302653e-01],
       [  9.81190389e-01,   8.48328902e-01,   1.42278607e-01],
       [  9.79643637e-01,   8.54866468e-01,   1.42453425e-01],
       [  9.77994918e-01,   8.61432314e-01,   1.42808191e-01],
       [  9.76264977e-01,   8.68015998e-01,   1.43350944e-01],
       [  9.74443038e-01,   8.74622194e-01,   1.44061156e-01],
       [  9.72530009e-01,   8.81250063e-01,   1.44922913e-01],
       [  9.70532932e-01,   8.87896125e-01,   1.45918663e-01],
       [  9.68443477e-01,   8.94563989e-01,   1.47014438e-01],
       [  9.66271225e-01,   9.01249365e-01,   1.48179639e-01],
       [  9.64021057e-01,   9.07950379e-01,   1.49370428e-01],
       [  9.61681481e-01,   9.14672479e-01,   1.50520343e-01],
       [  9.59275646e-01,   9.21406537e-01,   1.51566019e-01],
       [  9.56808068e-01,   9.28152065e-01,   1.52409489e-01],
       [  9.54286813e-01,   9.34907730e-01,   1.52921158e-01],
       [  9.51726083e-01,   9.41670605e-01,   1.52925363e-01],
       [  9.49150533e-01,   9.48434900e-01,   1.52177604e-01],
       [  9.46602270e-01,   9.55189860e-01,   1.50327944e-01],
       [  9.44151742e-01,   9.61916487e-01,   1.46860789e-01],
       [  9.41896120e-01,   9.68589814e-01,   1.40955606e-01],
       [  9.40015097e-01,   9.75158357e-01,   1.31325517e-01]];
   
if nargin < 1
    cm_data = cm;
else
    hsv=rgb2hsv(cm);
    hsv(153:end,1)=hsv(153:end,1)+1; % hardcoded
    cm_data=interp1(linspace(0,1,size(cm,1)),hsv,linspace(0,1,m));
    cm_data(cm_data(:,1)>1,1)=cm_data(cm_data(:,1)>1,1)-1;
    cm_data=hsv2rgb(cm_data);
  
end
end

