function [y, w, v, fwdModel, y_level1] = RunHybridHdcaClassifier(data, truth, trainingwindowlength, trainingwindowoffset, cvmode, level2data, fwdModelData)

% Run a hierarchical classifier with added "level 2" data.
%
% [y, w, v, fwdModel, y_level1] = RunHybridHdcaClassifier(data, truth, trainingwindowlength,
%    trainingwindowoffset, cvmode, level2data, fwdModelData)
%
% INPUTS:
% - data is a [DxTxN] matrix of data, where D is # channels, T is # time,
%   and N is # trials.
% - truth is a 1xN matrix of binary labels indicating the class of each
%   trial.
% - trainingwindowlength is a scalar indicating the number of samples that
%   should be in each training window.
% - trainingwindowoffset is an M-element vector indicating the offset of 
%   each training window in samples.
% - cvmode is a string indicating the cross-validation mode, used as input
%   to setGroupedCrossValidationStruct.m.
% - level2data is a NxQ matrix of non-time-dependent data for each trial.
%   That is, each column of level2data will receive a "temporal" weight as
%   if it were another output from the spatial weights.
% - fwdModelData is a [ExTxN] matrix (where E is # channels) used to
%   calculate the forward models. (e.g., if ICA activations are 'data', the
%   corresponding raw data could be used as 'fwdModelData').
%
% OUTPUTS:
% - y is a N-element vector, indicating the "interest score" for each
%   trial, a higher number indicating that the trial is more likely a target.
% - w is a DxMxP matrix, where D is the number of electrodes, M is the
%   number of windows, and P is the number of folds in cross-validation.
%   w(:,i,j) is the set of spatial weights found by the FLD that best 
%   discriminates the data in window i in the training trials of fold j.
% - v is a [1x(M+Q)xP] matrix, in which v(:,:,j) is the set of temporal
%   weights found by LR that best discriminates the data in the training
%   trials of fold j.
% - fwdModel is a DxMxP matrix, where D is the number of electrodes, M is 
%   the number of windows, and P is the # of folds in cross-validation.
%   fwdModel(:,i,j) is the forward model taken by multiplying the inverse 
%   of the y values from window i and fold j by all the data from window i.
% - y_level1 is an NxM matrix of the 'within-bin interest scores' for each bin
%   of the EEG.
%
% Created 5/29/13 by DJ based (heavily) on run_rsvp_classifier_rawdata.
% Updated 8/27/13 by DJ - added y_level1 output
% Updated 8/28/13 by DJ - added w_addOn==0 check
% Updated 12/6/13 by DJ - cleaned up code
% Updated 3/11/14 by DJ - added level2data input

% Handle defaults
if ~exist('level2data','var')
    level2data = [];
end
if ~exist('fwdModelData','var');
    fwdModelData = data;
end

% Declare cross-validation mode for evaluation set
cvmode2 = cvmode;
% Set up
nWindows = numel(trainingwindowoffset);
nWindowWeights = nWindows + size(level2data,2);
[nElecs, nSamples, nTrials] = size(data);
nFmElecs = size(fwdModelData,1); % for calculating forward model
cv = setCrossValidationStruct(cvmode,nTrials);
nFolds = cv.numFolds;
% Initialize
wCell = cell(1,nFolds);
vCell = cell(1,nFolds);
% Initialize weights y values
yTrain = cell(1,nFolds);
yEval = cell(1,nFolds);  
yTest = cell(1,nFolds);  
yTestTrials = cell(1,nFolds);
sampleTruth = repmat(reshape(truth,[1 1 nTrials]),[1 nSamples 1]);

% Set up logistic regression params
params.regularize=1;
params.lambda=1e1;
params.lambdasearch=0;
params.eigvalratio=1e-4;
params.vinit=zeros(nWindowWeights+1,1);
params.show = 0;
params.LOO = 0;    

% MAIN LOOP
for foldNum=1:nFolds       
    fprintf('Fold %d...',foldNum);
    tic;
    % Separate out training, evaluation, and testing data
    foldTrainingData = data(:,:,cv.incTrials{foldNum});
    foldTestingData = data(:,:,cv.valTrials{foldNum});
    % Make corresponding truth matrices
    foldTrainingTruth = sampleTruth(:,:,cv.incTrials{foldNum});        
    foldTrainingTruth_trials = foldTrainingTruth(1,1,:); % used later
    % Initialize cv struct & y values
    yTrain{foldNum} = nan(size(foldTrainingData,3),nWindows);
    yEval{foldNum} = nan(size(foldTrainingData,3),nWindows); 
    yTest{foldNum} = nan(size(foldTestingData,3),nWindows); 
    cv_fold = setCrossValidationStruct(cvmode2,length(cv.incTrials{foldNum}));
    % INNER LOOP
    for jFold = 1:cv_fold.numFolds
        innerTrainingData = foldTrainingData(:,:,cv_fold.incTrials{jFold});
        innerTestingData = foldTrainingData(:,:,cv_fold.valTrials{jFold});
        innerTrainingTruth = foldTrainingTruth(:,:,cv_fold.incTrials{jFold});
        for iWin=1:nWindows
            % Extract relevant data
            isInWin = trainingwindowoffset(iWin)+(0:trainingwindowlength-1);
            trainingData = permute(reshape(innerTrainingData(:,isInWin,:),size(innerTrainingData,1),length(isInWin)*size(innerTrainingData,3)), [2 1]);    
            testingData = permute(reshape(innerTestingData(:,isInWin,:),size(innerTestingData,1),length(isInWin)*size(innerTestingData,3)), [2 1]);    
            trainingTruth = permute(reshape(innerTrainingTruth(:,isInWin,:),size(innerTrainingTruth,1),length(isInWin)*size(innerTrainingTruth,3)), [2 1]);
            % Find spatial weights using FLD
            [~,~,~,~,coeff] = classify(testingData,trainingData,trainingTruth);
            wCell{foldNum}(:,iWin,jFold) = coeff(2).linear;
%             w(:,iWin,foldNum,jFold) = coeff(2).linear;
            % Get y values
            trainingAvg = permute(mean(innerTrainingData(:,isInWin,:),2), [3,1,2]);
            testingAvg = permute(mean(innerTestingData(:,isInWin,:),2), [3,1,2]);
            yTrain{foldNum}(cv_fold.incTrials{jFold},iWin) = trainingAvg*wCell{foldNum}(:,iWin,jFold);
            yEval{foldNum}(cv_fold.valTrials{jFold},iWin) = testingAvg*wCell{foldNum}(:,iWin,jFold);            
        end        
    end
    % use avg of wts across inner folds to find 'testing y values'
    for iWin = 1:nWindows
        isInWin = trainingwindowoffset(iWin)+(0:trainingwindowlength-1);
        testingAvg = permute(mean(foldTestingData(:,isInWin,:),2), [3,1,2]);
        yTest{foldNum}(:,iWin) = testingAvg*mean(wCell{foldNum}(:,iWin,:),3);
    end
      
    % Get add-on (level 2) data
    yTrain_AddOn = zeros(length(cv.incTrials{foldNum}),size(level2data,2));
    yTest_AddOn = zeros(length(cv.valTrials{foldNum}),size(level2data,2));
    for iAddOn = 1:size(level2data,2)
        if all(level2data(:,iAddOn)==1) % if this column is a simple offset...
            yTrain_AddOn(:,iAddOn) = 1; % ...then don't change it.
            yTest_AddOn(:,iAddOn) = 1;
        else
            trainingData_AddOn = level2data(cv.incTrials{foldNum},iAddOn);
            testingData_AddOn = level2data(cv.valTrials{foldNum},iAddOn);
            % Perform dummy classification to scale data properly
            [~,~,~,~,coeff] = classify(testingData_AddOn, trainingData_AddOn,foldTrainingTruth_trials(:));
            wAddOn = coeff(2).linear;
            if wAddOn==0 % Added for special case where class means are equal
                wAddOn = eps;
            end
            yTrain_AddOn(:,iAddOn) = trainingData_AddOn*wAddOn;
            yTest_AddOn(:,iAddOn) = testingData_AddOn*wAddOn;
        end
    end
    % Make Eval and Training Trials the same for Add-Ons (1D -> little overfitting)
    yEval_AddOn = yTrain_AddOn;
    
    % Append level2data to yEval and yTest
    yEvalData = cat(2,yEval{foldNum},yEval_AddOn);
    yTestData = cat(2,yTest{foldNum},yTest_AddOn);
      
    % Standardize stddev of each bin
    for iWin=1:nWindowWeights
        yTestData(:,iWin) = yTestData(:,iWin)/std(yEvalData(:,iWin)); % normalize using training data
        yEvalData(:,iWin) = yEvalData(:,iWin)/std(yEvalData(:,iWin));
    end
    
    % Run logistic regression to get temporal weights
    [~,~,stats] = RunSingleLR(permute(yEvalData,[2,3,1]),foldTrainingTruth_trials,params);
    % Extract temporal weights and find YIS interest scores
    vCell{foldNum} = stats.wts(1:nWindowWeights)';
    yTestTrials{foldNum} = vCell{foldNum}*yTestData';
    
    % Report elapsed time
    tRun = toc;
    fprintf('Done with fold %d! Took %.1f seconds.\n',foldNum,tRun);
    
end

% Re-format cell output of (par)for loop
nFolds2 = size(wCell{1},3); % inner folds
w = nan(nElecs,nWindows,nFolds2,nFolds);
v = nan(1,nWindowWeights,nFolds);
y = nan(1,nTrials);
y_level1 = nan(nTrials,nWindows);
for foldNum = 1:nFolds
    if ~isempty(wCell{foldNum})
        w(:,:,:,foldNum) = wCell{foldNum};
    end
    v(:,:,foldNum) = vCell{foldNum};    
    y(cv.valTrials{foldNum}) = yTestTrials{foldNum};
    y_level1(cv.valTrials{foldNum},:) = yTest{foldNum};
end

% Print Az
Az = rocarea(y,truth);
fprintf('%s Az value: %.3f\n',cvmode,Az);

% Get forward models
fwdModel = zeros(nFmElecs,nWindows,nFolds);
for foldNum=1:nFolds
    for iWin = 1:nWindows
        % Extract relevant data
        isInWin = trainingwindowoffset(iWin)+(0:trainingwindowlength-1);
        testData = data(:,isInWin,cv.valTrials{foldNum});
        fmData = fwdModelData(:,isInWin,cv.valTrials{foldNum});
        % Get y values
        dataAvg = permute(mean(testData,2), [3,1,2]);
        fmDataAvg = permute(mean(fmData,2), [3,1,2]);
        yAvg = dataAvg*mean(w(:,iWin,:,foldNum),3);
        % Compute forward model
        fwdModel(:,iWin,foldNum) = yAvg \ fmDataAvg;
    end
end

