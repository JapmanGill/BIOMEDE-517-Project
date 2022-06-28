function [foundOuts, unusedOuts] = vararginParser(varargin)
% This function parses a varargin input into another function. The order of
% foundOuts is the order of the inputs to this function varargin(2:end).
% The order of unusedOuts is the order of ocurrence of the unused inputs
% into the original function. The first element in this function's varargin
% should be varargin from the calling function. The remaining elements are
% the strings to match.

	vars = varargin{1}(1:2:end);											% get the list of input parameters from the original function
	foundOuts = cell(1, length(varargin)-1);								% allocate space for the parameters found
	unusedOuts = [];														% generate pointer for any input parameters from the original function that were not searched for
	
	for param = 1:length(vars)												% cycle through each of the original function's inputs
		argIdx = find(strcmpi(vars{param}, varargin(2:end)), 1, 'first');	% get the first index of that original function's input in the searched list of parameters
		if ~isempty(argIdx)													% if such an index exists
			foundOuts{argIdx} = varargin{1}{2*(param-1)+2};					% store it in the appropriate place in our output cell array
		else																% otherwise, if no index exists
			unusedOuts = [unusedOuts, {vars{param}}];						% the parameter is unused by the original function
		end
	end

end