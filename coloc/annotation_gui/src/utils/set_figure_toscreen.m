function set_figure_toscreen(varargin)
% set_figure_toscreen Multiple monitor figure control
% This function is useful in multiple monitor setups when you want to
% modify the figure behavior of new figures. It allows one to assign new
% figures to come up in the same position as a previously defined figure, or tell it
% which monitor to come up on as default.
%
% Usage: 
% set_figure_toscreen(a)
%    Where 'a' is numeric, either a monitor number (1, 2, 3, 4) or a figure
%    handle. If 'a' is monitor number, sets position to new monitor using
%    current defaults. If 'a' is a figure handle, then position is set to
%    exact same as the figure whose handle is used.
% set_figure_toscreen(opt,a)
%    Where opt is a string, either 'mon' to use a monitor, or 'fig' to
%    use a figure handle. a should be either a monitor number (if using
%    'mon') or a figure handle (if using 'fig').
%
% Created by Robert M. Flight, October 30, 2008. rflight79@gmail.com
% http://snipurl.com/rflight
% Modified RMF Nov 3, 2008
% Modified RMF May 18, 2009

nvarin = length(varargin); % how many variables coming in
monitors = get(0,'MonitorPositions');
nMonitor = size(monitors,1);

% Figure out what type of situation we have first
if isempty(varargin) % no input arguments, will use Matlab position to tell which monitor to use
    error('Must Give at least a Monitor Number or Figure Handle!');
elseif nvarin == 1 % check for monitor or figure handle
    loc = varargin{1};
    if isnumeric(loc) % only deal with numbers in this case
        allFig = findobj('Type','figure');
        isfig = allFig == loc;
        if sum(isfig) == 1 %if it matches a figure handle, use it, and can't match multiple figures
            locCase = 1;
        elseif (loc >= 1) && (loc <= nMonitor) % otherwise it should be one of the monitor designations
            locCase = 2;
        else
            error('Input is not a figure handle or monitor number!');
        end %if
    else
        error('Input should be numeric!');
    end %if
elseif nvarin == 2 % user supplied a type and the location
    locType = varargin{1}; 
    if ~ischar(locType), error('First argument should be type!'); end 
    loc = varargin{2};  
    if ~isnumeric(loc), error('Second argument should be location!'); end

    switch lower(locType)
        case 'fig' % if it is a figure handle, do this
            allFig = findobj('Type','figure');
            isfig = allFig == loc;
            if sum(isfig == 1)
                locCase = 1;
            else
                error('Figure handle not found!');
            end %if
        case 'mon' % if it is a monitor location, do this
            if (loc >= 1) && (loc <= nMonitor)
                locCase = 2;
            else
                error('Monitor Not Found!');
            end %if
        otherwise
            error('Unknown Option. Please use fig or mon!');
    end %switch
else
    error('Arguments Not Recognized! Doing Nothing ...');
end %if 

% now we know what type of situation the user is asking for, lets do it

% what is the current default position -> used for all cases
currDefault = get(0,'DefaultFigurePosition');

% first defines a monitor to put the figure on
% want to get the current default position, and move the figure to the same
% position on the other monitor
if (locCase == 2)
    
    currMonitor = 0;
    newMonitor = loc;
    
    % get the bottom and left positions for both monitors and figures
    monitorPos = monitors(:,1:2);
    figPos = currDefault(:,1:2);
    
    % subtract the monitor positions from figure positions
    difPos = (ones(nMonitor,1) * figPos) - monitorPos;
    
    % figure out if any have both positive differences
    isPlus = (difPos(:,1) >= 0) & (difPos(:,2) >= 0);
    
    % and calculate the actual distance between them
    distPos = sqrt(sum(difPos.^2,2));
    
    % how many positive distances are there
    nPlus = sum(isPlus);
    
    % which monitor are we currently on
    if nPlus == 1
        currMonitor = find(isPlus);
    elseif nPlus > 1
        whichPlus = find(isPlus);
        allPlus = distPos(isPlus);
        [mindist,minloc] = min(allPlus);
        currMonitor = whichPlus(minloc);
    else
        [mindist,minloc] = min(distPos);
        currMonitor = minloc;
    end %if
    
    % what is the distance from the corners
    currDist = difPos(currMonitor);
    
    newPosition = [monitors(newMonitor,1:2) + currDist currDefault(3:4)];
    


% now if using a figure handle to set the position
elseif locCase == 1
    newPosition = get(allFig(isfig),'Position');
end %if
    
% set the figure position if there were no errors above
set(0,'DefaultFigurePosition',newPosition);