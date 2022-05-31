function varargout = getZFeats(z, binsize, varargin)
% This function assembles continuous features from the z struct. This is
% similar to the FormatZContinuousFR.m function, but will extract all
% neural features and all behavior (true position and decode).
%	Inputs:		z:			The z struct we want to extract from.
%				binsize:	The binsize we want our data extracted in, in
%							ms.
%				featList:	(optional) A cell array of strings, where the
%							strings are the names of the fields of the z
%							struct we want to extract. This is used to only
%							extract a subset of the total information. The
%							default is to extract all neural features, all
%							behavior, and all decode.
%				lagMs:		(optional) A scalar to set the amount of lag in
%							ms between the neural activity and the
%							behavior, in case lags smaller than the binsize
%							are desired. The value here will delay the
%							neural activity relative to the behavior. This
%							input defaults to 0.
%				maMs:		(optional) An integer for how many milliseconds
%							to skip when computing moving averages of the
%							features. This option will compute a new
%							average value every maMs milliseconds, so for
%							example, if binsize=10 and maMs=2, then the
%							output will be computed as the average of 1:10
%							followed by 3:12, then 5:14, etc... Defaults to
%							0, which computes a new average every binsize
%							ms (boxcar windows).
%				bpf:		(optional) Only applicable if 'EMG' is entered
%							into 'featList'. This enables customization of
%							the filter parameters used to filter
%							synchronized raw data. It should be in the
%							order [low cutoff, high cutoff]. It defaults to
%							[100, 500].
%				bbfield:	(optional) The name of the field storing the
%							broadband data for 'EMG'. This defaults to
%							'CPDBB'.
%				notch:		(optional) This applies a 2nd-order notch
%							filter to broadband data. Not calling the
%							notch name-value pair at all will not enable
%							the notch filter (default). Including notch as
%							a name-value pair with an empty vector entered
%							as the value will result in [58, 62]Hz used for
%							the notch filter. Entering any other
%							two-element vector will use those elements as
%							the cutoffs for the filter.
%	Outputs:	feats:		A cell array containing all of the extracted
%							information. If featList is defaulted, it will
%							be in the order {'FingerAnglesTIMRL', 'Decode',
%							'Channel', 'NeuralFeature', 'TrialNumber'}.

	%% parse inputs
	
	validateScalar = @(n) isnumeric(n) && isscalar(n);
	
	vp = inputParser;
	vp.KeepUnmatched = true;
	vp.addRequired('z');
	vp.addRequired('binsize');
	vp.addParameter('featList', {'FingerAnglesTIMRL', 'Decode', 'Channel', 'NeuralFeature', 'TrialNumber'}, @(f) iscell(f) || ischar(f));
	vp.addParameter('lagMs', 0, validateScalar);
	vp.addParameter('maMs', 0, validateScalar);
	vp.addParameter('bpf', [100, 500], @(f) isequal(size(f), [1, 2]) || isequal(size(f), [2, 1]));
	vp.addParameter('bbfield', 'CPDBB', @(f) ischar(f));
	vp.addParameter('notch', [], @(f) isequal(size(f), [1, 2]) || isequal(size(f), [2, 1]) || isempty(f));
	
	vp.parse(z, binsize, varargin{:});
	
	unusedParams = fieldnames(vp.Unmatched);
	if ~isempty(unusedParams)
		unusedParams = sprintf('%s, ', unusedParams{:});
		warning(['The following parameters were unused: ', unusedParams(1:end-2)]);
	end
	
	featList = vp.Results.featList;
	lagMs = vp.Results.lagMs;
	maMs = vp.Results.maMs;
	bpf = vp.Results.bpf;
	bbfield = vp.Results.bbfield;
	notch = vp.Results.notch;
	
	if ~iscell(featList)
		featList = {featList};
	end
	
	if strcmpi(bbfield, 'CPDBB')
		timesField = 'CPDTimes';
	else
		timesField = 'CerebusTimes';
	end
	
	if ~any(strcmpi(vp.UsingDefaults, 'notch')) && isempty(notch)
		notch = [58, 62];
	end
	
	if nargout > 1 && nargout ~= length(featList)
		error('Number of output variables does not equal the number of requested features.');
	end
	
	feats = cell(1, length(featList));
	notConfigedWarn = [];
	notConfigedAndNotFieldWarn = [];

	%% extract the data
	
	if isfield(z, 'TrialSuccess')
		z = z(logical([z.TrialSuccess]));
	end
	if isfield(z, bbfield)
		
	end
	runidxs = diff([z.TrialNumber]);
	runidxs = [0, find(runidxs>1 | runidxs<=0), length(z)];
	for r = 2:length(runidxs)
		sampsToRm = ceil(lagMs/binsize);
		
		tstart = z(runidxs(r-1)+1).ExperimentTime(1);
		tstop = z(runidxs(r)).ExperimentTime(end);
		if maMs
			t1 = tstart:maMs:tstop-binsize;
		else
			t1 = tstart:binsize:tstop-binsize;
		end
		t2 = t1+binsize-1;
		
		t1St = t1-lagMs;
		t2St = t2-lagMs;
		t2St(t1St < tstart) = [];
		t1St(t1St < tstart) = [];
		
		etime = vertcat(z(runidxs(r-1)+1:runidxs(r)).ExperimentTime);
		etime = [-Inf, mean([etime(2:end)'; etime(1:end-1)']), +Inf];
		t1Ref = discretize(t1, etime)';
		t2Ref = discretize(t2, etime)';
		
		t1Nf = t1Ref-t1Ref(1)+1-lagMs;
		t2Nf = t2Ref-t1Ref(1)+1-lagMs;
		t2Nf(t1Nf < 1) = [];
		t1Nf(t1Nf < 1) = [];
		for field = 1:length(featList)
			if isfield(z, featList{field}) || strcmp(featList{field}, 'EMG')
				switch featList{field}
					case {'Channel', 'CPDChannel', 'SingleUnit', 'SingleUnitHash'}
						feat = arrayfun(@(c) dumbAf(z(runidxs(r-1)+1:runidxs(r)), c, featList{field}), 1:length(z(1).(featList{field})), 'UN', false);
						feats{field} = [feats{field}; cell2mat(arrayfun(@(c) overlapHC(feat{c}, t1St, t2St), 1:length(feat), 'UN', false)')'];
					case 'NeuralFeature'
						feat = cell2mat(arrayfun(@(x) x.NeuralFeature, z(runidxs(r-1)+1:runidxs(r)), 'UN', false)');
						sampwidth = cell2mat(arrayfun(@(x) x.SampleWidth, z(runidxs(r-1)+1:runidxs(r)), 'UN', false)');
						feats{field} = [feats{field}; cell2mat(arrayfun(@(x1,x2) sum(feat(x1:x2,:),1) / sum(sampwidth(x1:x2)), t1Nf, t2Nf,'UN',false))];
					case {'TrialNumber', 'TargetPos', 'TargetScaling', 'ClosedLoop'}
						feat = cell2mat(arrayfun(@(x) repmat(x.(featList{field}), [length(x.ExperimentTime), 1]), z(runidxs(r-1)+1:runidxs(r)), 'UN', false)');
						times = round(mean([t1Ref-t1Ref(1)+1, t2Ref-t1Ref(1)+1], 2));
% 						feat = arrayfun(@(x1,x2) round(mean(feat(x1:x2))), t1Ref-t1Ref(1)+1, t2Ref-t1Ref(1)+1);
						feats{field} = [feats{field}; feat(times, :)];
					case 'FingerAnglesTIMRL'
						feat = cell2mat(arrayfun(@(x) x.(featList{field}), z(runidxs(r-1)+1:runidxs(r)), 'UN', false)');
						feat = cell2mat(arrayfun(@(x1,x2) mean(feat(x1:x2, :), 1), t1Ref-t1Ref(1)+1, t2Ref-t1Ref(1)+1, 'UN', false));
						feat = [feat, [diff(feat, 1, 1); zeros(1, size(feat, 2))], [diff(feat, 2, 1); zeros(2, size(feat, 2))]];
						feats{field} = [feats{field}; feat(sampsToRm+1:end, :)];
					case 'EMG'
						if all(~isfield(z, {bbfield, timesField}))
							error([bbfield, ' and/or ', timesField, ' are not fields in the z struct!']);
						end
						
						% get the sampling rate for this block of data
						totSampsBb = sum(arrayfun(@(x) size(z(x).(bbfield), 2), runidxs(r-1)+1:runidxs(r)));
						totSampsXpc = sum(arrayfun(@(x) length(z(x).ExperimentTime), runidxs(r-1)+1:runidxs(r)));
						srate = totSampsBb/totSampsXpc;
						
						% we need to determine how many samples were
						% skipped between the end of one trial and the
						% beginning of the next
						timeDiff = [0, arrayfun(@(x, y) z(x).(timesField)(1) - z(y).(timesField)(end) - 1, runidxs(r-1)+2:runidxs(r), runidxs(r-1)+1:runidxs(r)-1)];	% subtract an extra one: t starts at 105, t-1 ends at 100, there are 4 samples missing, not 5
						timeDiff = cumsum(timeDiff);
						
						sampTimes = cell2mat(arrayfun(@(x, y) z(x).(timesField) - y, runidxs(r-1)+1:runidxs(r), timeDiff, 'UN', false));
						sampTimes = ceil(double(sampTimes - sampTimes(1) + 1) / (mean(diff(sampTimes))/srate));	% reference the data to itself, where each element of sampTimes corresponds to 1ms, and scale by the sampling rate
						
						[b, a] = butter(2, bpf/(mean(diff(sampTimes))*500), 'bandpass');	% this uses an estimate of the true sampling rate assuming xPC is keeping true time
						feat = cell2mat(arrayfun(@(x) x.(bbfield)', z(runidxs(r-1)+1:runidxs(r)), 'UN', false)');
						feat = filter(b, a, feat);
						
						if ~isempty(notch)	% optional notch filter
							[b, a] = butter(2, notch/(mean(diff(sampTimes))*500), 'stop');
							feat = filter(b, a, feat);
						end
						
						feats{field} = [feats{field}; cell2mat(arrayfun(@(x1,x2) sum(abs(feat(x1:x2, :)), 1) / (x2 - x1 + 1), sampTimes(t1Nf), sampTimes(t2Nf+1)-1, 'UN', false)')];
					case 'Decode'
						feat = cell2mat(arrayfun(@(x) x.(featList{field}), z(runidxs(r-1)+1:runidxs(r)), 'UN', false)');
						feat = cell2mat(arrayfun(@(x) feat(x, :), t1Ref-t1Ref(1)+1, 'UN', false));
						feats{field} = [feats{field}; feat(sampsToRm+1:end, :)];
					otherwise
						notConfigedWarn = [notConfigedWarn, field];
						feat = cell2mat(arrayfun(@(x) x.(featList{field}), z(runidxs(r-1)+1:runidxs(r)), 'UN', false)');
						feat = cell2mat(arrayfun(@(x1,x2) mean(feat(x1:x2, :), 1), t1Ref-t1Ref(1)+1, t2Ref-t1Ref(1)+1, 'UN', false));
						feats{field} = [feats{field}; feat(sampsToRm+1:end, :)];
				end
			else
				notConfigedAndNotFieldWarn = [notConfigedAndNotFieldWarn, field];
			end
		end
	end
	
	if ~isempty(notConfigedWarn)
		for w = unique(notConfigedWarn)
			warning([featList{w}, ' was not configured for feature extraction.']);
		end
	end
	
	if ~isempty(notConfigedAndNotFieldWarn)
		for w = unique(notConfigedAndNotFieldWarn)
			warning([featList{w}, ' was not configured for feature extraction and is not a field in the z struct.']);
		end
	end
	
	if nargout == length(featList)
		varargout = feats;
	else
		varargout = {feats};
	end
end

function out = overlapHC(X, edgeMin, edgeMax)
	out = zeros(size(edgeMin));
	for i = 1:length(edgeMin)
		out(i) = sum(X<=edgeMax(i) & X>=edgeMin(i));
	end
end

function out = dumbAf(z, c, fn)	% Af for arrayfun, because Matlab doesn't allow indexable expressions, which is dumb af
	tmp = cell(1, length(z));
	for t = 1:length(z)
		tmp{t} = single(z(t).(fn)(c).SpikeTimes);
	end
	isem = cellfun(@isempty, tmp);
	out = cell2mat(tmp(~isem)');
end