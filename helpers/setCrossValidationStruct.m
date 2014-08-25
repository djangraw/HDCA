function cv = setCrossValidationStruct(cvmode,nTrials)

% Automatically separates a set number of trials into multi-fold groups.
%
% cv = setGroupedCrossValidationStruct(cvmode,nTrials)
% 
% INPUTS:
% -cvmode is a string indicating the type of cross-validation to be
% performed. The options are 'nocrossval' (1 fold),'loo' (leave one out),
% or '<x>fold', where x is a whole number.
% -nTrials is a scalar indicating the number of trials in your data.
%
% OUTPUTS:
% -cv is a struct with fields incTrials, outTrials, valTrials, and
% numFolds.
%
% Created 5/22/13 by DJ based on setGroupCrossValidationStruct.m by BC.
% Updated 8/24/13 by DJ - changed incTrials lines to remove setdiff (speed)

cv = [];
cv.mode = cvmode;

if strcmp(cvmode,'nocrossval') % all trials are included in training, none for testing.
    cv.numFolds = 1;
    cv.incTrials = cell(1);
    cv.outTrials = cell(1);
    cv.valTrials = cell(1);
    cv.incTrials{1} = 1:nTrials;
    cv.outTrials{1} = [];
    cv.valTrials{1} = 1:nTrials; % test on the same data you trained on
elseif strcmp(cvmode,'loo') % leave one trial out for training, use it for testing
    cv.numFolds = nTrials;
    cv.incTrials = cell(1,cv.numFolds);
    cv.outTrials = cell(1,cv.numFolds);
    cv.valTrials = cell(1,cv.numFolds);
    for j=1:nTrials
%         cv.incTrials{j} = setdiff(1:nTrials,j);
        cv.incTrials{j} = [1:j-1, j+1:nTrials];
        cv.outTrials{j} = j;
        cv.valTrials{j} = j;
    end
elseif strcmp(cvmode((end-3):end),'fold') % separate data into folds, use one fold for testing and the rest for training
    cv.numFolds = str2double(cvmode(1:(end-4)));
    cv.incTrials = cell(1,cv.numFolds);
    cv.outTrials = cell(1,cv.numFolds);
    cv.valTrials = cell(1,cv.numFolds);
    
    % Split the data into roughly equally-sized folds
    foldSizes = floor(nTrials/cv.numFolds)*ones(1,cv.numFolds);
    foldSizes(1:(nTrials-foldSizes(1)*cv.numFolds)) = foldSizes(1)+1;
       
    eInd = 0;  % index of end of this fold
	for j=1:cv.numFolds
		sInd = eInd + 1; % index of start of this fold
		eInd = sInd + foldSizes(j) - 1; % index of end of this fold
        sset = sInd:eInd; % indices in this set
				
		cv.outTrials{j} = sset;
        cv.valTrials{j} = cv.outTrials{j};
%         cv.incTrials{j} = setdiff(1:nTrials,cv.outTrials{j});
        cv.incTrials{j} = [1:sInd-1, eInd+1:nTrials];        
	end
else
	error('Unknown cross-validation mode');
end
