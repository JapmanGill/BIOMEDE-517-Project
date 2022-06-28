function [adjX, adjY] = adjustFeats(X, Y, varargin)
% This function takes in neural data X and behavior Y and returns two
% "adjusted" neural data and behavior matrices based on the settings in
% varargin. Specifically, the amount of lag between the neural data and
% behavior can be set in units bins and the number of historical bins of
% neural data can be set.
%	Inputs:		X:		The neural data, which should be [t, neu] in size,
%						where t is the number of samples and neu is the
%						number of neurons.
%				Y:		The behavioral data, which should be [t, dim] in
%						size, where t is the number of samples and dim is
%						the number of behavioral dimensions.
%				lag:	(optional) The number of bins to lag the neural
%						data relative to the behavioral data. For example,
%						adjustFeats(X, Y, 'lag', 1) will return X(1:end-1)
%						for adjX and Y(2:end) for adjY. This defaults to 0.
%				hist:	(optional) The number of bins to append to each
%						sample of neural data from the previous 'hist'
%						bins. This defaults to 0.
%				fillB4:	(optional) This fills previous neural data with
%						values before the experiment began. A single scalar
%						will fill all previous neural data with that value.
%						Otherwise, a vector equal to the first dimension of
%						X (number of neurons) should represent the value to
%						fill for each channel. This defaults to 0, which
%						disables the option.
%	Outputs:	adjX:	The adjusted neural data.
%				adjY:	The adjusted behavioral data.


	if (nargin < 2)
		error('Need both neural and behavioral data inputs.');
	end
	
	[foundParams, unusedParams] = vararginParser(varargin, 'lag', 'hist', 'fillB4');	% this function is in Sam's utility code folder
	
	if ~isempty(unusedParams)
		unusedParams = sprintf('%s, ', unusedParams{:});
		warning(['The following parameters were unused: ', unusedParams(1:end-2)]);
	end
	
	[lag, hist, fillB4] = deal(foundParams{:});
	
	if isempty(lag)
		lag = 0;
	end
	
	if isempty(hist)
		hist = 0;
	end
	
	nNeurons = size(X, 2);
	
	% reshape neural data to include historical bins
	if ~isempty(fillB4)
		if isscalar(fillB4)
			X = [fillB4*ones(hist, nNeurons); X];
			Y = [zeros(hist, size(Y, 2)); Y];
		else
			X = [repmat(fillB4, [hist, 1]); X];
			Y = [zeros(hist, size(Y, 2)); Y];
		end
	end
	adjX = zeros(size(X, 1)-hist, nNeurons*(hist+1));
	for samp = hist+1:size(X, 1)
		adjX(samp - hist, :) = reshape(X(samp - hist:samp, :)', [1, nNeurons * (hist+1)]);
	end
	adjY = Y(hist+1:end, :);
	
	% now offset the neural data and behavior
	adjX = adjX(1:end-lag, :);
	adjY = adjY(lag+1:end, :);

end