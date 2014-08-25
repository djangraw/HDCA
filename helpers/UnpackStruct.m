function UnpackStruct(myStruct)

% Takes every field in the given struct and makes it its own variable.
%
% UnpackStruct(myStruct)
%
% INPUTS:
% -myStruct is a struct whose fields you want to make into variables.
%
% OUTPUTS:
% -variables will be created in the function that called this one.
%
% Created 9/21/11 by DJ.

fields = fieldnames(myStruct);

for i=1:numel(fields)
%     if evalin('caller',sprintf('exist(''%s'',''var'')',fields{i})) % if the calling function already has a var by this name
%         warning('UnpackStruct:Overwrite','Overwriting variable %s',fields{i}); % warn the user
%     end
    assignin('caller',fields{i},myStruct.(fields{i})); % send variable to calling function
end