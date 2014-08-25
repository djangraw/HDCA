function PlotHybridHdcaClassifier(fwdModel, v, chanlocs, vtimes_ms)

% Plot scalp maps and temporal weights from hybrid classifier output.
%
% PlotHybridHdcaClassifier(fwdModels, v, chanlocs, vtimes_ms)
%
% NOTE: This function is made to work with the output of
% RunHybridHdcaClassifier and plot the mean across folds, but it could also
% be used to plot the mean across subjects. In this case, the first 2 
% inputs should be changed to be the mean across folds for each subject 
% concatenated in the 3rd dimension. 
%
% INPUTS:
% - fwdModel is a DxMxP matrix, where D is the number of electrodes, M is 
%   the number of windows, and P is the # of folds in cross-validation.
%   fwdModel(:,i,j) is the forward model taken by multiplying the inverse 
%   of the y values from window i and fold j by all the data from window i.
% - v is a [1x(M+Q)xP] matrix, where Q is the number of 'level 2 data' 
%   features. v(:,:,j) is the set of temporal weights found by LR that best 
%   discriminates the data in the training trials of fold j.
% - chanlocs is a D-element vector of structs from the chanlocs field of
%   the EEGLAB struct used to train the classifier.
% - vtimes_ms is an (M+Q)-element vector of the times (in ms) at which each
%   value of v should be plotted. The first M values will be interpreted as
%   EEG data, and the rest will be plotted separately as 'other data'.
%
% Created 3/13/14 by DJ.


%% Plot average forward models

% Set up
clf;
cmax = max(max(abs(mean(fwdModel,3))));
% cmax = 10;
nBins = size(fwdModel,2);
nCols = ceil(nBins/2);

% Plot scalp maps
for i=1:nBins % for each time bin
    h=subplot(4,nCols,rem(i-1,2)*nCols+floor((i-1)/2)+1);
    % shift left or right by 1/4 the figure width
    pos = get(h,'position');
    if mod(i,2)==1        
        set(h,'position',[pos(1)-pos(3)/4, pos(2:4)]);
    else
        set(h,'position',[pos(1)+pos(3)/4, pos(2:4)]);
    end
    topoplot(mean(fwdModel(:,i,:),3),chanlocs,'maplimits',[-cmax cmax],'conv','on'); % do include mullet
    title(sprintf('%g ms',vtimes_ms(i)));
end
% Create one colorbar for all plots
axes('Position',[.88 .62 .1 .24],'CLim',[-cmax cmax],'visible','off');
colorbar;

% Annotate figure with fwd models title
h = MakeFigureTitle('Forward Models (Mean Across Folds) (uV)');
set(h,'fontweight','normal')


% Plot temporal weights
subplot(2,1,2); hold on;
% plot EEG weights
errorbar(vtimes_ms(1:nBins),mean(v(:,1:nBins,:),3),std(v(:,1:nBins,:),[],3)/sqrt(size(v,3)),...
    'b.-','linewidth',2,'markersize',10);           
% Plot level 2 weights
errorbar(vtimes_ms(nBins+1:end),mean(v(:,nBins+1:end,:),3),std(v(:,nBins+1:end,:),[],3)/sqrt(size(v,3)),...
    'g.-','linewidth',2,'markersize',10);           
legend('EEG Features','Other Features')

% Annotate plot
set(gca,'xgrid','on','box','on')
title('Temporal Weights (Mean +/- stderr across folds)');
ylabel('temporal weights')
xlabel('time of bin center (ms)')
hold on
plot(get(gca,'xlim'),[0 0],'k--');
