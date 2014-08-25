function h = MakeFigureTitle(title,plotOnAxes)

% Places a specified title on the titlebar and, if desired, at the top of 
% the current figure.  Returns a handle to the text.
%
% h = MakeFigureTitle(title,plotOnAxes)
%
% INPUTS:
% -title is the string you want printed on the titlebar of the figure.
% -plotOnAxes is a binary value indicating whether you want to plot the
% specified title on the figure itself (if so, it will make a new set of
% axes at the top of the figure that are invisible except for this text).
% NOTE: thes new axes will becomes the current ones - you must specify a 
% different set of axes before plotting anything new.
%
% OUTPUTS:
% -h is a handle to the text plotted at the top of the figure (if
% plotOnAxes==1. Otherwise, h=[]).
%
% Created by DJ 10/19/07.
% Updated 9/1/09 by DJ.
% Updated 12/29/11 by DJ - added plotOnAxes option
% Updated 7/13/12 by DJ - comments


% Handle inputs
if nargin<2 || isempty(plotOnAxes)
    plotOnAxes = 1;
end

% Set figure name
set(gcf,'NumberTitle','off','Name',title)     

% Plot text on the figure itself
if plotOnAxes % if desired
    axes('Position', [.5 .98 .02 .02], 'Visible', 'off') % make new, invisible axes
    h = text(0,0,show_symbols(title),'horizontalalignment','center',... 
        'fontweight','bold'); % plot text in bold
else 
    h=[]; % if the figure text is not desired, provide empty output
end