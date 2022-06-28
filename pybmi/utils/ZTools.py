import os
import pdb
import numpy as np
import scipy.io as sio
import h5py
import re
import copy
import glob
import sys
import pandas as pd
import warnings     # for raising warnings if user requested non-configured or missing features
from scipy import signal    # for Butterworth filter


# TODO Maybe eventually find a way to make everything the same class within the zarray
# have to make global for pickle to work
# recreates scipy's way of organizing structs. used to be called mat_struct_sim, renamed to mat_struct_sim
class mat_struct_sim(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, np.asarray([mat_struct_sim(x) if isinstance(x, dict) else x for x in b]))
            else:
                if isinstance(b, dict):
                    setattr(self, a, mat_struct_sim(b))
                elif isinstance(b, np.ndarray):
                    setattr(self, a, b.squeeze())  # deals with issues where scipy squeezes but mat_struct_sim does not
                else:
                    setattr(self, a, b)
        setattr(self, '_fieldnames', [*d])

def ZStructTranslator(direc, overwrite=False, use_py=False, numChans=96):
    '''
    This function translates the File Logger binaries into dictionaries. Most of the work is directly taken from Scott
    Ensel's ZStructTranslator class, created in 2019-2020

    Inputs:	direc:		- the directory from which to grab the data
           	overwrite:	Default: does NOT overwrite previous zstruct in folder
           				- False: to load any previous zStruct in the folder, if available
                          - True: to read .bin files and overwrite or save z struct file
                          - if overwrite is true use_py argument does not matter
              use_py:     Default: loads in .mat first
                          - False: loads in .mat file
                          - True: loads in .npy file
              numChans:   Default: 96 channels
                          - number of channels recorded on
    Returns: z:         - the zstruct in zarray format. Can be treated with all the normal operations used on object np.ndarrays,
                            plus some extra functions

    NOTE: Make sure the folder path includes the entire directory structure
          with  (i.e. the name\date\run folders).
    NOTE: This script assumes that neural data is sent as variable length
          packets with an EndPacket byte of 255. It also assumes that there
          is exactly 1 feature in each neural packet.
    Based on ZStructTranslator.m written by Zach Irwin

    Additional Notes (Not sure if still relevant, may change):
    IF USING A MAC YOU MUST STORE FILES LOCALLY
    CANNOT ACCESS FILES LOCATED ON SMB DRIVE

    FOR PC PLS REPLACE EVERY \ WITH \\ in the file path

    example use:

    import ZStructTranslator

    direc = 'Z:\\Data\\RPNI\\P1\\Chronic\\16th Visit\\2018-06-19\\Run-009'

    EMG_data = ZStructTranslator(direc, overwrite=True)
    #this will create a new .npy file
    # overwrite = False to load a .mat file
    # overwrite = False and use_py = True to load a .npy file

    ############ How to access the data #################
    # EMG_data is an object

    EMG_data[0].FingerAnglesTIMRL
    # this accesses the first trial of the data
    # to get all of the trials
    for trial in EMG_data[0:]:
    print(trial.FingerAnglesTIMRL)

    # the fields of FingerAnglesTIMRL can be accessed like a normal array FingerAnglesTIMRL[:,finger]

    ########### For Spiking data ##############
    for trial in EMG_data[:10]:
    for numchan in range(len(trial.Channel)):
    print(trial.Channel[numchan].SpikeTimes)

    ########## How to see the fields ###########
    EMG_data = ZStructTranslator(direc, overwrite=True)
    print(EMG_data[0]._fieldnames)
    '''

    # loops over multiple characters to put into a single string
    def int_to_string(in_array):
        final_string = []
        for number in range(len(in_array)):
            final_string.append(chr(in_array[number]))

        final_string = ''.join(final_string)

        return final_string

    # generator for the paths of the HDF5 file
    # adapted from https://stackoverflow.com/questions/51548551/reading-nested-h5-group-into-numpy-array
    def h5py_dataset_iterator(g, prefix='', skiprefs=True):
        for key in g.keys():
            if skiprefs and key == '#refs#': #refs is big and has a lot of keys and we dont need it so skip it if you want.
                continue
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    def traverse_datasets(hdf_file):
        with h5py.File(hdf_file, 'r') as f:
            for path, _ in h5py_dataset_iterator(f):
                yield path

    # function begins here
    path = os.path.normpath(direc)  # this is system independent splitting of path
    folders = path.split(os.sep) # this is system independent splitting of path

    # check for previous zStruct
    zfilename = "Z_" + folders[-3] + "_" + folders[-2] + "_" + folders[-1] + ".mat"  # this is full .mat filename
    zfilename_python = "Z_" + folders[-3] + "_" + folders[-2] + "_" + folders[-1] + ".npy" #this is full .npy filename

    zFullPath = os.path.join(direc, zfilename)
    zFullPath_py = os.path.join(direc, zfilename_python)

    # if loading data in checks to see if previous .mat or .npy file exists

    if (not overwrite and os.path.isfile(zFullPath) and not use_py):  # loads in .mat files by default
        print('Loading previous .mat Z Struct...')
        try:  # if the existing .mat file is in a non-HDF5 file (i.e. non -v7.3 .mat file)
            z = sio.loadmat(zFullPath, struct_as_record=False, squeeze_me=True)['z']
            z = zarray(z)
            print('Finished Loading previous .mat Z Struct')
        except NotImplementedError:
            dataset = [] #list for the dataset
            names = [] #list of all the names
            j = 0
            with h5py.File(zFullPath, 'r') as f: # opens the file
                for dset in traverse_datasets(zFullPath): #uses function to loop through nested sections
                    if "/z/" in dset:
                        names.append(dset.replace("/z/", "")) # this gets all the field names
                        dataset.append([]) #might be able to get rid of this line to reduce if statement
                        data = f.get(dset)
                        for i in range(len(data[:])): #this part derefrences the HDF5 so you can see the data
                            st = data[i][0]
                            obj = f[st]
                            try: # this is becuase on neural data Channel becomes an HDF5 group and not a HDF5 dataset so this throw an error
                                dataset[j].append(obj[:].T)  # .T is so its in same form as sio
                            except AttributeError: # this only happens on Neural data (due to SpikeTimes being nested)
                                dataset[j].append([])
                                for k in range(len(obj['SpikeTimes'])):
                                    st2 = obj['SpikeTimes'][k][0]
                                    obj2 = f[st2]
                                    if np.array_equal(obj2[:],np.asarray([0, 0])): # for some reasons it shows [] as [0,0] so replace them
                                        dataset[j][i].append([])
                                    else:
                                        dataset[j][i].append(obj2[:]) # otherwise format is fine
                        j += 1

                dict_data = []
                z = []
                num_trials = len(dataset[0])
                for j in range(num_trials): #this creates a dictionary for each trial to match scipy.io format
                    all_data = {} #intializes dictionary
                    #the next lines are to make this indexed the same as scipy.io
                    for i in range(len(names)):  #for scalar values gets rid of lists of lists format
                        # this is all formatting for the spike times since its nested and makes this hard
                        if names[i] == 'Channel':
                            all_data[names[i]] = []
                            for k in range(numChans):
                                if len(dataset[i][j][k]) == 1: # if the index is [x] not []
                                    if len(dataset[i][j][k][0]) != 1: #if value is of form [ x x x x x x . . . ]
                                        all_data[names[i]].append({'SpikeTimes': dataset[i][j][k][0].astype(int)}) #if an array get rid of one list
                                    else:
                                        all_data[names[i]].append({'SpikeTimes': dataset[i][j][k][0][0].astype(int)}) #if just a scalar value get rid of both lists
                                else:
                                    all_data[names[i]].append({'SpikeTimes': dataset[i][j][k]}) # if just empty [] leave it

                        else:
                            if len(dataset[i][j]) == 1: # if value is of form [[...]]  and not [[...][...][...][...]]
                                if len(dataset[i][j][0]) != 1: #if value is of form [ x x x x x x . . . ]
                                    all_data[names[i]] = dataset[i][j][0] #if an array get rid of one list
                                else:
                                    all_data[names[i]] = dataset[i][j][0][0] #if just a scalar value get rid of both lists
                            else:
                                all_data[names[i]] = dataset[i][j] #if in the form [[...][...][...][...]] keep it that wasy

                    dict_data.append(all_data)
                    # this turns the dictionary into a object
                    z.append(mat_struct_sim(dict_data[j]))
                z = zarray(z) #has to be an ndarray to match scipy output.
                print('Finished Loading previous .mat Z Struct')
    else:
        if (not overwrite and os.path.isfile(zFullPath_py) and use_py):
            print('Loading previous .npy Z struct . . . ')

            z = np.load(zFullPath_py, allow_pickle=True).tolist()
            z = zarray(z)
            print('Finished loading .npy file')
            return z

        else:
            try:
                # load in and read z translator
                f = open(r'{}'.format(os.path.join(direc, "zScript.txt")), "r")
                if f.mode == 'r':
                    contents = f.read()
            except:
                print("zScript.txt file not found. Make sure you're in correct folder")
                sys.exit(1)

            if not overwrite and not os.path.isfile(zFullPath) and not use_py:
                print("{} not found. Please check directory".format(zfilename))
                if (os.path.isfile(zFullPath_py)):
                    print('Found .npy version, loading instead.')
                    print('Loading previous .npy Z struct . . . ')

                    z = np.load(zFullPath_py, allow_pickle=True).tolist()
                    z = zarray(z)
                    print('Finished loading .npy file')
                    return z

                print("Creating .npy instead..")
            elif not overwrite and not os.path.isfile(zFullPath_py) and use_py:
                print("{} not found. Please check directory".format(zfilename_python))
            print("Overwriting Previous .npy Z Struct or no .npy file found. Creating New one . . .")

            # supported data types and their byte sizes
            cls = {'uint8': 1, 'int8': 1, 'uint16': 2, 'int16': 2, 'uint32': 4, 'int32': 4, 'single': 4, 'double': 8}
            # data types and their python equivalent
            data_con = {'uint8': np.ubyte, 'int8': np.byte, 'uint16': np.ushort, 'int16': np.short, 'uint32': np.uintc,
                        'int32': np.intc, 'single': np.single, 'double': np.double}
            # Split Z string into its P/M/D/N substrings:
            zstr = re.split(':.:', contents) #1st entry is blank

            # Split each substring into its fieldname/datatype/numels substrings:
            for i in range(len(zstr)):
                zstr[i] = zstr[i].split('-') #last entry in each cell is blank

            #Collect names, types, and sizes into list of list
            names = []
            types = []
            sizes = []
            for i in range(1, len(zstr)):
                names.append([])
                types.append([])
                sizes.append([])
                for j in range(0, len(zstr[i]) - 1, 3):
                    names[i - 1].append(zstr[i][j])
                for j in range(1, len(zstr[i]) - 1, 3):
                    types[i - 1].append(zstr[i][j])
                for j in range(2, len(zstr[i]) - 1, 3):
                    sizes[i - 1].append(zstr[i][j])

            # Set flag(s) for specific field formatting:
            for i in range(len(names)):
                for j in range(len(names[i])):
                    if names[i][j] == 'SpikeChans':
                        spikeformat = True
                    else:
                        spikeformat = False

            #Recover number of fields in each file type:
            fnum = []
            for i in range(len(names)):
                fnum.append(len(names[i])) # Number of fields in each file

            fnames = [None] * 2 * sum(fnum)
            # use ord() to change hexidecimal notation to correct int values
            bsizes = copy.deepcopy(sizes)
            for i in range(len(fnum)):
                for j in range(len(names[i])):
                    bsizes[i][j] = ord(bsizes[i][j])

            # Calculate byte sizes for each feature and collect field names:
            m = 0
            for i in range(len(fnum)):
                for j in range(len(names[i])):
                    fnames[m] = names[i][j]
                    m += 2
                    # calculate the bytes sizes
                    bsizes[i][j] = cls[types[i][j]] * bsizes[i][j] #Match type to cls, get type byte size, multiply by feature length:

            # Calculate bytes per timestep for each file:
            bytes = []
            for i in range(len(bsizes)):
                bytes.append(int(np.sum(bsizes[i]) + 2)) #plus 2 for each trial count

            # Get number of trials in this run:
            ###################
            trial_list = glob.glob(os.path.join(direc, 'tParams*'))
            ntrials = len(trial_list)
            trials = []
            for i in range(ntrials):
                trials.append(int(re.findall('\d+', trial_list[i])[-1]))
            trials = np.sort(trials)
            if trials[-1] != ntrials:
                warnings.warn("There is at least 1 dropped trial")

            # initalize the dictionary with correct field names
            dict_data = []
            for j in range(trials[-1]):  # this creates a dictionary for each trial
                all_data = {}
                for i in range(0, len(fnames), 2):
                    all_data[fnames[i]] = None
                all_data['TrialNumber'] = None
                if spikeformat: # this is so its a nested dictionary and can match other formats
                    all_data['NeuralData'] = None
                    all_data['Channel'] = []
                    for k in range(numChans):
                        all_data['Channel'].append({'SpikeTimes': []})
                dict_data.append(all_data)

            ############################## Parse Data Strings Into Dictionary: ######################################
            data = [[], [], [], []] #initilize data
            dropped_list = []
            for i in range(trials[-1]):
                try:
                    # add trial number to dict
                    dict_data[i]['TrialNumber'] = i + 1

                    #read in data files
                    data[0] = np.fromfile(os.path.join(direc, 'tParams{}.bin'.format(i + 1)), dtype='uint8')
                    data[1] = np.fromfile(os.path.join(direc, 'mBehavior{}.bin'.format(i + 1)), dtype='uint8')
                    data[2] = np.fromfile(os.path.join(direc, 'dBehavior{}.bin'.format(i + 1)), dtype='uint8')
                    data[3] = np.fromfile(os.path.join(direc, 'neural{}.bin'.format(i + 1)), dtype='uint8')

                except:
                    dict_data[i]['TrialNumber'] = None # this will set up the removal of empty dictionaries
                    dropped_list.append(i) # sets up the spike formatting
                    print("Trail Number {} was dropped".format(i+1))
                    continue # this skips to the next trial

                # Iterate through file types 1-3 and add data to Z:
                for j in range(4 - spikeformat):

                    # Calculate # of timesteps in this file:
                    nstep = int(len(data[j]) / bytes[j])

                    # Calculate the byte offsets for each feature in the timestep:
                    offs = (3 + np.cumsum(bsizes[j])).tolist() #cumsum only works on np arrays so convert to list after
                    offs.insert(0, 3) # starts at 3 because of trail counts

                    # Iterate through each field:
                    for k in range(fnum[j]):
                        # Create a byte mask for the uint8 data:
                        bmask = np.zeros(bytes[j], dtype=np.uint8)
                        bmask[range(offs[k] - 1, offs[k] + bsizes[j][k] - 1)] = 1
                        bmask = np.matlib.repmat(bmask, 1, nstep)
                        bmask = bmask[0]

                        #Extract data and cast to desired type:
                        dat = data[j][bmask == 1].view((data_con[types[j][k]]))  # this has to be types

                        #Reshape the data and add to dict
                        dict_data[i][names[j][k]] = np.reshape(dat, (nstep, -1))

                        #format the data so scalars are not in lists, arrays are in lists, and multiple arrays are lists of lists
                        if len(dict_data[i][names[j][k]]) == 1: # if value is of form [[...]]  and not [[...][...][...][...]]
                            if len(dict_data[i][names[j][k]][0]) != 1:
                                dict_data[i][names[j][k]] = dict_data[i][names[j][k]][0]
                            else:
                                dict_data[i][names[j][k]] = dict_data[i][names[j][k]][0][0]

                # Extract Neural data packets (split around TrialCount and End Packet byte):
                if spikeformat:
                    new_string = int_to_string(data[3]) # convert to one continuous string
                    try:
                        # If it is one of these special characters python doesnt like it and understand that its not special
                        # so add the \ to escape the special character
                        # for some reason though | thinks its super special
                        # if you just add the backslash it then grabs that when matching and messes up
                        # so the second if statement is removing that character from each match the first time it appears
                        # and that works but I have no idea why this is such a problem
                        if chr((i+1)%256) in ['.', '*', '^', '$', '+', '?', '{', '}', '[', ']', '|', '(', ')' ,'\\']:
                            ndata = re.findall(
                                '\\' + int_to_string(np.asarray([np.ushort(i + 1)]).view(np.ubyte)) + '[^ÿ]*ÿ',
                                new_string)
                        else:
                            ndata = re.findall(int_to_string(np.asarray([np.ushort(i+1)]).view(np.ubyte)) + '[^ÿ]*ÿ', new_string)

                        neural_data = []
                        for m in range(len(ndata)):
                            if len(ndata[m][2:-1]) == 0:
                                neural_data.append([])
                            elif len(ndata[m][2:-1]) == 1:
                                neural_data.append([ord(ndata[m][2:-1])])
                            else:
                                neural_data_2 = []
                                for n in range(len(ndata[m][2:-1])):
                                    neural_data_2.append(ord(ndata[m][2:-1][n]))
                                neural_data.append(neural_data_2)

                        dict_data[i]['NeuralData'] = neural_data

                    except re.error: # data is an empty cell
                        dict_data[i]['NeuralData'] = []

            # Format Specific Fields
            ############################################################################
            # Change neural data field into spike times per channel:
            if spikeformat:
                for i in range(trials[-1]):
                    if i in dropped_list: # if i equals the value of a dropped trial then skip it
                        continue

                    spikenums = np.zeros((numChans, len(dict_data[i]['NeuralData'])))
                    for t in range(len(dict_data[i]['NeuralData'])):
                        for j in range(len(dict_data[i]['NeuralData'][t])):
                            if dict_data[i]['NeuralData'][t][j] != 0:
                                spikenums[dict_data[i]['NeuralData'][t][j] - 1, t] += 1

                    for c in range(numChans):
                        if np.any(spikenums[c, :]):
                            times = dict_data[i]['ExperimentTime'][spikenums[c, :] == 1]
                            spikenumsi = spikenums[c, (spikenums[c, :] == 1)]
                            idx = np.cumsum(spikenumsi)
                            j = np.ones((1, int(idx[-1])), dtype=int)

                            dict_data[i]['Channel'][c]['SpikeTimes'] = times[np.cumsum(j) - 1].T

                        if len(dict_data[i]['Channel'][c]['SpikeTimes']) == 1:
                            if len(dict_data[i]['Channel'][c]['SpikeTimes'][0]) != 1:
                                dict_data[i]['Channel'][c]['SpikeTimes'] = dict_data[i]['Channel'][c]['SpikeTimes'][0].astype(
                                    int)
                            else:
                                dict_data[i]['Channel'][c]['SpikeTimes'] = dict_data[i]['Channel'][c]['SpikeTimes'][0][
                                    0].astype(int)

                    #Removes these two fields
                    dict_data[i].pop('SpikeChans', None)
                    dict_data[i].pop('NeuralData', None)


            # this next one removes the skipped trials
            # must use a generator because of indexing issues if you don't
            if trials[-1] != ntrials: # only run if there is a trial that is dropped to save time
                dict_data = [trial for trial in dict_data if trial['TrialNumber'] != None]

            # convert from dict to object
            z = []
            for i in range(ntrials):
                z.append(mat_struct_sim(dict_data[i]))
            z = zarray(z)
            # save file
            print('Saving now. . .')
            # I put this try excpet because I was getting weird problems from the training of KF in the terminal
            try:
                np.save(zFullPath_py, z)
            except:
                np.save(zFullPath_py, z)

            print('New ZStruct saved at {} \nFile is named {}'.format(direc, zfilename_python))
    return z

# Adapted from JCostello's PrintZStruct function that prints a z struct to the output (located in PrintZStruct.py).
def zStructToPandas(zstruct):
    """
    Loads a z struct into a Pandas DataFrame object.
    :param zstruct: The z struct to be converted to a DataFrame
    :return: df - The DataFrame version of the z struct
    """
    # load into dataframe
    return zstruct.asdataframe()

# Taken from getZFeats.m:
# a helper function that calculates the quantity of spike times in between two numbers, edgeMin[i] and edgeMax[i]
def overlapHC(X,
        edgeMin,
        edgeMax):
    """
    Calculates the quantity of spike times in between two numbers, edgeMin[i] and edgeMax[i]
    :param X: The feature input
    :param edgeMin: The minimum edges for each index
    :param edgeMax: The maximum edges for each index
    :return: numSpikes - The number of spikes at each index
    """
    numEdges = len(edgeMin)
    numSamples = len(X)
    numSpikes = np.zeros(numEdges)
    for i in range(numSamples):
        numSpikes = np.add(numSpikes, np.logical_and(np.less_equal(X[i], edgeMax), np.greater_equal(X[i], edgeMin)))
    return numSpikes

def getZFeats(z,
              binsize,
              featList=['FingerAnglesTIMRL', 'Decode', 'Channel', 'NeuralFeature', 'TrialNumber'],
              lagMs=0,
              maMs=0,
              bpf=[100, 500],
              bbfield='CPDBB',
              notch=[],
              trimBlankTrials=True,
              removeFirstTrial=True):
    """
    Extracts continuous features from a given z struct.
    :param z: The z struct to extract features from (a zarray or zDataframe).
    :param binsize: The bin size in which to extract the data, given in milliseconds (ms).
    :param featList: (optional) A string array that contains the names of the fields of the z struct to extract.
                     Default is ['FingerAnglesTIMRL', 'Decode', 'Channel', 'NeuralFeature', 'TrialNumber'].
    :param lagMs: (optional) The amount of lag, in ms, between the neural activity and the behavior. Delays the neural
                  activity relative to the behavior. Default is 0.
    :param maMs: (optional) The amount of time, in ms, to skip when computing moving averages of the features. A new
                 average is computed every 'maMs' ms. Default is 0.
    :param bpf: (optional) Applicable for 'EMG' features. Used for customizing filter parameters in filtering
                synchronizing raw data. Its form is defined as [low cutoff, high cutoff]. Default is [100, 500].
    :param bbfield: (optional) Name of field storing the broadband data for 'EMG'. Default is 'CPDBB'.
    :param notch: (optional) Applies a 2nd-order notch filter to broadband data. By default, the notch filter is
                  disabled.
    :param trimBlankTrials: (optional) Determines if trials marked with BlankTrial as True (1) will be automatically
                            trimmed before processing z struct. Default is True (blank trials will be trimmed).
    :param removeFirstTrial: (optional) Determines if the first trial should be removed from the z struct. The first
                             trial is often a throwaway trial used to set up each run, so the default is True.
    :return: feats - an array containing the corresponding features. Each feature is stored at an index in the dict
                     corresponding to the feature's name. For example, the features for 'FingerAnglesTIMRL' would be
                     stored at feats['FingerAnglesTIMRL'].
    """
    ## Check input arguments
    # Determine if notch filter will be used -> default is []

    # Convert arrays to numpy arrays
    featList = np.asarray(featList)
    bpf = np.asarray(bpf)
    notch = np.asarray(notch)

    # Validate inputs
    # binsize must be a numeric scalar
    if not np.isscalar(binsize) or not (type(binsize) == int or type(binsize) == float):
        raise Exception('binsize must be a numeric scalar!')
    # lagMs must be a numeric scalar
    if not np.isscalar(lagMs) or not (type(lagMs) == int or type(lagMs) == float):
        raise Exception('lagMs must be a numeric scalar!')
    # maMs must be a numeric scalar
    if not np.isscalar(maMs) or not (type(maMs) == int or type(maMs) == float):
        raise Exception('maMs must be a numeric scalar!')
    # bpf must be an array of size (2,1), (1,2), (2,)
    if not (bpf.shape == (2,) or bpf.shape == (2, 1) or bpf.shape == (1, 2)):
        raise Exception('bpf must be empty, a 2 x 1, or 1 x 2 array!')
    # notch must be an empty array or array of size (2,1), (1,2), (2,)
    if not (notch.shape == (2,) or notch.shape == (2, 1) or notch.shape == (1, 2) or notch.shape == (0,)):
        raise Exception('notch must be empty, a 2 x 1, or 1 x 2 array!')
    # bbfield must be string
    if not isinstance(bbfield, str):
        raise Exception('bbfield must be a string!')
    # featList must be an array of strings
    if featList.shape == '()':
        raise Exception('featList must be an array of strings!')
    else:
        try:
            for fL in featList:
                if not isinstance(fL, str):
                    raise Exception('featList must be an array of strings!')
        except:
            raise Exception('featList must be an array of strings!')
    # trimBlankTrials must be a boolean
    try:
        trimBlankTrials = bool(trimBlankTrials)
    except:
        raise Exception('trimBlankTrials must be a boolean!')
    # removeFirstTrial must be a boolean
    try:
        removeFirstTrial = bool(removeFirstTrial)
    except:
        raise Exception('removeFirstTrial must be a boolean!')

    # ------------------------------------------ Extract data ---------------------------------------------------------
    # Convert the z struct to a pandas DataFrame type
    if isinstance(z, pd.DataFrame):
        zStructDF = z
    else:
        zStructDF = z.asdataframe()

    # Remove first trial if desired
    if removeFirstTrial:
        zStructDF = zStructDF[1:]

    # Trim out blank trials if desired
    if trimBlankTrials:
        zStructDF = zStructDF.loc[zStructDF['BlankTrial'] == 0]

    # Filter the z struct by only good trials if the 'GoodTrial' column exists
    if 'GoodTrial' in zStructDF.columns:
        zStructDF = zStructDF.loc[zStructDF['GoodTrial'] == 1]

    # Determine timesField based on bbfield
    timesField = 'CerebusTimes'
    if bbfield.upper() == 'CPDBB':
        timesField = 'CPDTimes'

    # Initialize the dictionary that will hold each feature
    returnedFeatures = dict.fromkeys(featList)  # initialized as None

    # Holds warnings for not configured and non-field requests
    notConfiguredWarning = []
    notConfiguredAndNotFieldWarning = []

    # Get indices of the current run. Each value in runIndices represents the index of the next run to start or stop at.
    # For example, if runIndices = [0, 413], then we will take all runs from index 0 to 413.
    # This is important if we are skipping any runs, such as runs that are not good trials.
    # First, extract the trial numbers from the z struct
    trialNumbers = zStructDF['TrialNumber'].to_numpy()
    # We want to subtract each trial number by 1 because indexing in Python starts at 0 compared to 1 for MATLAB.
    trialNumbers = trialNumbers - 1
    # Extract the experiment times to be used later
    experimentTimes = zStructDF['ExperimentTime'].to_numpy()
    # Find differences between each trial number to check if any trials are being skipped or repeated
    trialDifferences = np.diff(trialNumbers)
    # Determine if any trials are being skipped or repeated by finding trial differences that are > 1 or <= 0
    changedTrials = np.nonzero(np.logical_or(trialDifferences > 1, trialDifferences <= 0))
    initialStartIndex = -1     # start at -1 because we will add 1 to the start for each window
    # Append the changedTrials after the startIndex into runIndices
    runIndices = np.append(np.array([initialStartIndex]), changedTrials)
    # Add final run to indices
    runIndices = np.append(runIndices, len(zStructDF) - 1)
    # runIndices now holds the start and stop indices for each run!

    # If some of the runIndices are not consecutive, then we will have more than one window of runs!
    # For this reason, we will loop through each run window. For example, suppose that we use runs 0 to 100, skip 101,
    # and then use 102 to 200. We would then have two separate run windows --- the first being from 0 to 100 and the
    # second being from 102 to 200.
    for runWindow in range(len(runIndices) - 1):
        samplesToRemove = int(np.ceil(lagMs / binsize))
        # Calculate the start and stop times
        startIndex = runIndices[runWindow] + 1  # add 1 to move to the next window
        # Start time is the first time in the start index
        startTime = experimentTimes[startIndex][0]
        stopIndex = runIndices[runWindow + 1]
        # Stop time is the last time in the stop index
        stopTime = experimentTimes[stopIndex][-1]

        # Depending on the value of maMs, we can determine t1 and t2
        # Refer to conversion from MATLAB here: https://www.mathworks.com/help/matlab/ref/colon.html
        desiredLastPoint = stopTime - binsize
        numPoints = int(np.fix((desiredLastPoint - startTime) / binsize))
        actualLastPoint = startTime + numPoints * binsize
        # t1 is the start time of each index
        t1 = np.linspace(startTime, actualLastPoint, numPoints+1)
        if maMs:
            numPoints = int(np.fix((desiredLastPoint - startTime) / maMs))
            actualLastPoint = startTime + numPoints * maMs
            t1 = np.linspace(startTime, actualLastPoint, numPoints+1)
        # t2 is the end time of each index
        t2 = t1 + binsize - 1

        # Offset t1Start and t2Start by lagMs
        t1Start = t1 - lagMs
        t2Start = t2 - lagMs
        # Remove values that are less than startTime (and keep values that are greater than or equal to start time)
        t2Start = t2Start[t1Start >= startTime]
        t1Start = t1Start[t1Start >= startTime]

        # Determine eTime by vertically concatenating all runs in the current run window together
        # (runIndices[runWindow+1]+1 is the end index because a Python array takes array[startIndex:endIndex+1]
        allExperimentTimes = np.concatenate(experimentTimes[startIndex:stopIndex+1])
        # Offset experiment times by 1 forward and 1 backward
        forwardOffsetTime = allExperimentTimes[1:]  # 1 to end
        backwardOffsetTime = allExperimentTimes[:-1]    # 0 to (end - 1)
        # Stack forward offsets above backward offsets
        stackedOffsets = np.stack((forwardOffsetTime, backwardOffsetTime))
        # Take mean time at each index (along the columns)
        meanOffsets = np.mean(stackedOffsets, axis=0)
        # Digitize times into bins
        t1Digitized = np.digitize(t1, bins=meanOffsets)
        t2Digitized = np.digitize(t2, bins=meanOffsets)
        # Take lagMs into account
        t1DigitizedLag = t1Digitized - t1Digitized[0] - lagMs
        t2DigitizedLag = t2Digitized - t1Digitized[0] - lagMs
        # Remove values that are less than 0 (and keep the rest that are greater than or equal to 0)
        t2DigitizedLag = t2DigitizedLag[t1DigitizedLag >= 0]
        t1DigitizedLag = t1DigitizedLag[t1DigitizedLag >= 0]

        # Loop through each field in the feature list (featList)
        # This is converted from the switch structure in MATLAB: https://www.mathworks.com/help/matlab/ref/switch.html
        # For a cell array case_expression, at least one of the elements of the cell array matches switch_expression
        # For example, for case {'pie', 'pie3'}, and input of 'pie3' would trigger this case
        # When a case expression is true, MATLAB executes the corresponding statements and exits the switch block.
        # The MATLAB break statement ends execution of a for or while loop, but does not end execution of a switch
        # statement. This behavior is different than the behavior of break and switch in C.
        for currentFeature in featList:
            # Check if the current feature is a column in the z struct OR is equal to 'EMG'
            if currentFeature in zStructDF.columns or currentFeature == 'EMG':
                # Extract features based on the current feature
                if currentFeature in ['Channel', 'SingleUnit', 'SingleUnitHash']:   # Extracting spiking rates
                    # Get the first spikeTimes value from each z struct index from start to stop
                    spikeTimesRaw = zStructDF[currentFeature][startIndex:(stopIndex+1)]
                    spikeTimes = []
                    for trial in spikeTimesRaw:
                        spikeTrial = []
                        # Loop through each channel in trial
                        for channel in trial:
                            spike = getattr(channel, 'SpikeTimes')
                            spikeTrial.append(spike)
                        spikeTimes.append(spikeTrial)
                    spikeTimesDF = pd.DataFrame(data=spikeTimes)
                    # For each channel c, extract the cth element from each trial as long as the element is NOT empty
                    numChannels = spikeTimesDF.shape[1]
                    # Spike numbers matrix, where each column represents a new channel
                    spikeNumbersFeat = np.zeros([len(t1Start), numChannels])
                    for c in range(numChannels):
                        currentChannelFeatures = []
                        # Extract current column
                        currentColumn = spikeTimesDF[c]
                        # Remove values in current column that are empty
                        for r in range(len(currentColumn)):
                            currentValue = currentColumn[r]
                            # Add each element individually if the currentValue is an array
                            if isinstance(currentValue, np.ndarray) and len(currentValue) > 0:    # a non-empty array
                                for val in currentValue:
                                    currentChannelFeatures.append(val)
                            elif not isinstance(currentValue, np.ndarray):  # is NOT an empty array (is an integer)
                                currentChannelFeatures.append(currentValue)
                        # Calculate number of spikes
                        spikeNumbersFeat[:, c] = overlapHC(currentChannelFeatures, t1Start, t2Start)
                    # Append spike numbers feat to the feature dictionary
                    currentFeatDict = returnedFeatures[currentFeature]
                    if currentFeatDict is None:
                        currentFeatDict = spikeNumbersFeat
                    else:
                        currentFeatDict = np.append(currentFeatDict, spikeNumbersFeat, axis=0)
                    returnedFeatures[currentFeature] = currentFeatDict

                elif currentFeature == 'NeuralFeature':     # Spiking band power
                    # Extract NeuralFeature from dataframe for start and stop index
                    neuralFeatures = zStructDF[currentFeature].to_numpy()[startIndex:(stopIndex+1)]
                    # Concatenate into one matrix
                    neuralFeatures = np.concatenate(neuralFeatures)
                    # Extract sample width from dataframe
                    sampleWidth = zStructDF['SampleWidth'].to_numpy()[startIndex:(stopIndex+1)]
                    # Concatenate into one matrix
                    sampleWidth = np.concatenate(sampleWidth)
                    # For both t1DigitizedLag and t2DigitizedLag, calculate the spiking band power, given by
                    # sum(feat(x1:x2,:),1) / sum(sampwidth(x1:x2))
                    numSamples = t1DigitizedLag.shape[0]
                    spikingBandPower = np.zeros([numSamples, neuralFeatures.shape[1]])
                    for i in range(numSamples):
                        t1_i = t1DigitizedLag[i]
                        t2_i = t2DigitizedLag[i]
                        # Generate indices between x1_i and x2_i
                        indices_i = np.linspace(t1_i, t2_i, abs(t2_i - t1_i) + 1, dtype=int)
                        spikingBandPower[i, :] = np.sum(neuralFeatures[indices_i, :], axis=0) / np.sum(sampleWidth[indices_i])
                    # Add spiking band power to the returned feats
                    # First, check if this feature is already initialized -> if not, initialize it!
                    currentFeatDict = returnedFeatures[currentFeature]
                    if currentFeatDict is None:    # this feature's array in the dictionary has not been initialized yet
                        currentFeatDict = spikingBandPower
                    else:   # this feature is already initialized so we will append the new features as new rows
                        currentFeatDict = np.append(currentFeatDict, spikingBandPower, axis=0)
                    returnedFeatures[currentFeature] = currentFeatDict

                # case for extracting trial info repeated to match the length of the trial according to binsize
                elif currentFeature in ['TrialNumber', 'TargetPos', 'TargetScaling', 'ClosedLoop']:
                    # Get trial numbers from start to stop index
                    trialFeats = zStructDF[currentFeature].to_numpy()[startIndex:(stopIndex+1)]
                    # Get experiment times from start to stop index
                    experimentTimesStartStop = experimentTimes[startIndex:(stopIndex+1)]
                    experimentLength = len(experimentTimesStartStop)
                    # Loop through each experiment and add the times to the array
                    repeatedFeats = np.array([])
                    for i in range(experimentLength):
                        currentExperimentLength = len(experimentTimesStartStop[i])
                        repeatedArray = np.matlib.repmat(trialFeats[i], currentExperimentLength, 1)
                        if repeatedFeats.size == 0:
                            repeatedFeats = repeatedArray
                        else:
                            repeatedFeats = np.append(repeatedFeats, repeatedArray, axis=0)
                    # Get times
                    t1DigitizedOffset = t1Digitized - t1Digitized[0]
                    t2DigitizedOffset = t2Digitized - t1Digitized[0]
                    t1And2DigitizedOffset = np.hstack((t1DigitizedOffset.reshape(-1, 1), t2DigitizedOffset.reshape(-1, 1)))
                    # In MATLAB, a number with a decimal of 0.5 is rounded up to the nearest integer.
                    # However, in NumPy (and Python), a number with decimal of 0.5 is rounded to the nearest even number
                    # 24.5 is rounded to 25 in MATLAB but 24 in Python
                    # In order to preserve the MATLAB rounding behavior in Python, we can round using
                    # y = int(np.floor(n + 0.5))
                    # https://stackoverflow.com/questions/28617841/rounding-to-nearest-int-with-numpy-rint-not-consistent-for-5
                    times = np.floor(np.mean(t1And2DigitizedOffset, axis=1) + 0.5).astype(int)
                    # Extract the features at the times intervals
                    trialFeats = repeatedFeats[times]
                    # Append trialFeats to returnedFeats
                    currentFeatDict = returnedFeatures[currentFeature]
                    if currentFeatDict is None:
                        currentFeatDict = trialFeats
                    else:
                        currentFeatDict = np.append(currentFeatDict, trialFeats, axis=0)
                    returnedFeatures[currentFeature] = currentFeatDict

                elif currentFeature == 'FingerAnglesTIMRL':
                    # For the current z indices, extract the finger angles
                    fingerAngles = zStructDF[currentFeature].to_numpy()[startIndex:(stopIndex+1)]
                    # Concatenate fingerAngles into one matrix
                    fingerAngles = np.concatenate(fingerAngles)
                    # Take the mean of each column
                    x1 = t1Digitized - t1Digitized[0]
                    x2 = t2Digitized - t1Digitized[0]
                    # https://stackoverflow.com/questions/29297633/behavior-of-colon-operator-with-matrix-or-vector-arguments
                    # combinedIndices = x1:x2
                    # For each index i, get the features between x1[i] and x2[i], and take their mean
                    numSamples = x1.shape[0]
                    fingerFeats = np.zeros([numSamples, fingerAngles.shape[1]])
                    for i in range(len(x1)):
                        x1_i = x1[i]
                        x2_i = x2[i]
                        # Generate indices between x1_i and x2_i
                        indices_i = np.linspace(x1_i, x2_i, abs(x2_i-x1_i)+1, dtype=int)
                        # Extract the mean finger angles for these indices
                        fingerFeats[i, :] = np.mean(fingerAngles[indices_i, :], axis=0)
                    # Append the extra matrix to fingerFeats
                    featsFirstDiff = np.diff(fingerFeats, n=1, axis=0)
                    featsSecondDiff = np.diff(fingerFeats, n=2, axis=0)
                    # Rows of zeros for padding
                    oneRowZeros = np.zeros([1, fingerFeats.shape[1]])
                    twoRowZeros = np.zeros([2, fingerFeats.shape[1]])
                    # Construct first and second difference matrices
                    firstDiffMatrix = np.concatenate((featsFirstDiff, oneRowZeros), axis=0)
                    secondDiffMatrix = np.concatenate((featsSecondDiff, twoRowZeros), axis=0)
                    # Construct difference matrix
                    diffMatrix = np.concatenate((firstDiffMatrix, secondDiffMatrix), axis=1)
                    # Concatenate diffMatrix to the right side of fingerFeats
                    fingerFeats = np.concatenate((fingerFeats, diffMatrix), axis=1)
                    # Remove samples based on lagMs
                    fingerFeats = fingerFeats[samplesToRemove:, :]
                    # Append to returned dictionary
                    currentFeatDict = returnedFeatures[currentFeature]
                    if currentFeatDict is None:
                        currentFeatDict = fingerFeats
                    else:
                        currentFeatDict = np.append(currentFeatDict, fingerFeats, axis=0)
                    returnedFeatures[currentFeature] = currentFeatDict

                elif currentFeature == 'EMG':
                    # Check if bbField and timesField are fields in the z struct
                    if bbfield not in zStructDF.columns or timesField not in zStructDF.columns:
                        # Throw error
                        raise Exception(bbfield + ' and/or ' + timesField + ' are not fields in the z struct!')
                    # Get sampling rate for block of data
                    # Extract bbfield for this block
                    bbFieldBlock = zStructDF[bbfield].to_numpy()[startIndex:(stopIndex+1)]
                    # Loop through each bbField
                    totalSamplesBB = 0
                    for bbF in bbFieldBlock:
                        sizeField = bbF.shape[1]
                        totalSamplesBB += sizeField
                    # Get total samples XPC
                    experimentTimesBlock = experimentTimes[startIndex:(stopIndex+1)]
                    totalSamplesXPC = 0
                    for et in experimentTimesBlock:
                        lengthExperiment = len(et)
                        totalSamplesXPC += lengthExperiment
                    # Calculate sampling rate
                    samplingRate = totalSamplesBB / totalSamplesXPC
                    # Determine how many samples were skipped between the end of one trial and the beginning of the next
                    # x = runidxs(r-1)+2:runidxs(r)
                    timesFieldArray = zStructDF[timesField].to_numpy()
                    xStartIndex = startIndex + 1
                    xStopIndex = stopIndex
                    xIndices = np.linspace(xStartIndex, xStopIndex, abs(xStopIndex - xStartIndex) + 1).astype(int)
                    yStartIndex = startIndex
                    yStopIndex = stopIndex - 1
                    yIndices = np.linspace(yStartIndex, yStopIndex, abs(yStopIndex - yStartIndex) + 1).astype(int)
                    lengthIndices = len(xIndices)
                    timeDifference = np.zeros(lengthIndices)
                    for i in range(lengthIndices):
                        timeDifference[i] = timesFieldArray[xIndices[i]][0] - timesFieldArray[yIndices[i]][-1] - 1
                    # Insert 0 at beginning of time difference and get cumulative sum
                    timeDifference = np.insert(timeDifference, 0, 0)
                    timeDifference = np.cumsum(timeDifference)
                    # Get sample times
                    sampleTimes = timesFieldArray[startIndex:(stopIndex+1)] - timeDifference
                    sampleTimes = np.concatenate(sampleTimes)
                    # Reference data to itself, where each element of sampleTimes corresponds to 1ms, and scale by the
                    # sampling rate
                    sampleRateScale = np.mean(np.diff(sampleTimes)) / samplingRate
                    refSamples = (sampleTimes - sampleTimes[0]).astype(float)
                    sampleTimes = np.ceil(refSamples / sampleRateScale)
                    # Apply butterworth filter
                    # Uses an estimate of the true sampling rate assuming xPC is keeping true time
                    criticalFrequency = bpf / (np.mean(np.diff(sampleTimes)) * 500)
                    filterOrder = 2
                    b, a = signal.butter(filterOrder, criticalFrequency, btype='bandpass', output='ba')
                    # Extract feature with filter
                    emgFeat = bbFieldBlock
                    # Concatenate together
                    emgFeat = np.concatenate(emgFeat, axis=1).T
                    # scipy filter operates on the last dimension, we must specify axis=0 to match the MATLAB result
                    # https://stackoverflow.com/questions/16936558/matlab-filter-not-compatible-with-python-lfilter?rq=1
                    emgFeat = signal.lfilter(b, a, emgFeat, axis=0)
                    # Apply optional notch filter
                    if notch.size != 0:   # notch is not empty
                        b, a = signal.butter(2, notch / (np.mean(np.diff(sampleTimes)) * 500), btype='stop')
                        emgFeat = signal.lfilter(b, a, emgFeat, axis=0)
                    # Extract feature
                    x1 = (sampleTimes[t1DigitizedLag]).astype(int)
                    x2 = (sampleTimes[t2DigitizedLag+1]-1).astype(int)
                    numSamples = len(x1)
                    extractedEMGFeatures = np.zeros([numSamples, emgFeat.shape[1]])
                    for i in range(numSamples):
                        x1_i = x1[i]
                        x2_i = x2[i]
                        currentIndices = np.linspace(x1_i, x2_i, x2_i - x1_i + 1).astype(int)
                        sampledFeat = np.sum(np.abs(emgFeat[currentIndices, :]), axis=0) / (x2_i - x1_i + 1)
                        extractedEMGFeatures[i, :] = sampledFeat
                    # Add EMG features to feature dictionary
                    currentFeatDict = returnedFeatures[currentFeature]
                    if currentFeatDict is None:
                        currentFeatDict = extractedEMGFeatures
                    else:
                        currentFeatDict = np.append(currentFeatDict, extractedEMGFeatures, axis=0)
                    returnedFeatures[currentFeature] = currentFeatDict

                elif currentFeature == 'Decode':
                    # Extract 'Decode' from dataframe
                    decodeValues = zStructDF[currentFeature].to_numpy()[startIndex:(stopIndex+1)]
                    # Concatenate decodeValues into one matrix
                    decodeValues = np.concatenate(decodeValues)
                    # Get x1 to feed into the decode value calculation
                    x1 = t1Digitized - t1Digitized[0]
                    # For each element of x1, extract the feature at that row
                    decodeValues = decodeValues[x1, :]
                    # Remove samples if applicable
                    decodeValues = decodeValues[samplesToRemove:, :]
                    # Append decoded values to returned features
                    currentFeatDict = returnedFeatures[currentFeature]
                    if currentFeatDict is None:
                        currentFeatDict = decodeValues
                    else:
                        currentFeatDict = np.append(currentFeatDict, decodeValues, axis=0)
                    returnedFeatures[currentFeature] = currentFeatDict

                # If a feature is requested but not configured, assume it is captured at 1ms resolution and should be
                # averaged in bin size
                else:   # otherwise
                    # Add status to not configured array
                    notConfiguredWarning.append(currentFeature)
                    # Get feature in the current interval from start to stop
                    featureInterval = zStructDF[currentFeature][startIndex:(stopIndex+1)].to_numpy()
                    x1 = t1Digitized - t1Digitized[0]
                    x2 = t2Digitized - t1Digitized[0]
                    numSamples = len(x1)
                    # Check if array can be indexed
                    totalNumIndices = 0
                    mustRepeatValues = False
                    for fi in featureInterval:
                        # Check if value is scalar or array
                        if hasattr(fi, 'shape') and fi.ndim > 0:    # is array
                            totalNumIndices += np.max(fi.shape)
                        else:
                            mustRepeatValues = True
                            break
                    if totalNumIndices < np.max(x2):
                        mustRepeatValues = True
                    resultingFeats = []
                    if mustRepeatValues:
                        numColumns = 1
                        if featureInterval.ndim > 1:
                            numColumns = featureInterval.shape[1]
                        resultingFeats = np.zeros([numSamples, numColumns])
                        # Repeat values in feature array to rescale to t1Digitized (as was done in 'TrialNumber')
                        # Get experiment times from start to stop index
                        experimentTimesStartStop = experimentTimes[startIndex:(stopIndex + 1)]
                        experimentLength = len(experimentTimesStartStop)
                        # Loop through each experiment and add the times to the array
                        repeatedFeats = np.array([])
                        for i in range(experimentLength):
                            currentExperimentLength = len(experimentTimesStartStop[i])
                            repeatedArray = np.matlib.repmat(featureInterval[i], currentExperimentLength, numColumns)
                            repeatedFeats = np.append(repeatedFeats, repeatedArray)
                        # Get times
                        t1DigitizedOffset = t1Digitized - t1Digitized[0]
                        t2DigitizedOffset = t2Digitized - t1Digitized[0]
                        t1And2DigitizedOffset = np.hstack((t1DigitizedOffset.reshape(-1, 1), t2DigitizedOffset.reshape(-1, 1)))
                        times = np.floor(np.mean(t1And2DigitizedOffset, axis=1) + 0.5).astype(int)
                        # Extract the features at the times intervals
                        resultingFeats = repeatedFeats[times]
                    else:   # we can take the mean normally (with re-sizing array)
                        numColumns = 1
                        if featureInterval[0].ndim > 1:
                            numColumns = featureInterval[0].shape[1]
                        resultingFeats = np.zeros([numSamples, numColumns])
                        # Concatenate values into one large array
                        featureInterval = np.concatenate(featureInterval)
                        # Take the mean value
                        for samp in range(numSamples):
                            x1_samp = x1[samp]
                            x2_samp = x2[samp]
                            current_i = np.linspace(x1_samp, x2_samp, x2_samp - x1_samp + 1).astype(int)
                            resultingFeats[samp, :] = np.mean(featureInterval[current_i], axis=0)
                        # Remove samples if needed
                        resultingFeats = resultingFeats[samplesToRemove:, :]
                    # Append resultingFeats to the existing feature
                    currentFeatDict = returnedFeatures[currentFeature]
                    if currentFeatDict is None:
                        currentFeatDict = resultingFeats
                    else:
                        currentFeatDict = np.append(currentFeatDict, resultingFeats, axis=0)
                    returnedFeatures[currentFeature] = currentFeatDict

            else:   # current feature is not in z struct and is not 'EMG'
                # Add status to not configured and not field array
                notConfiguredAndNotFieldWarning.append(currentFeature)

    # Notify user if some requested features were not configured or not available
    # Not configured (but available in the z struct)
    if notConfiguredWarning:    # list is NOT empty
        # Get unique features so that we don't print out any features more than once
        uniqueFeats = np.unique(notConfiguredWarning)
        # Loop through each feature
        for unConfigFeat in uniqueFeats:
            warnings.warn(str(unConfigFeat) + " was not configured for feature extraction.")
    # Not configured and not available in the z struct
    if notConfiguredAndNotFieldWarning:    # list is NOT empty
        # Get unique features so that we don't print out any features more than once
        uniqueFeats = np.unique(notConfiguredAndNotFieldWarning)
        # Loop through each feature
        for unConfigFeat in uniqueFeats:
            warnings.warn(str(unConfigFeat) + " was not configured for feature extraction and is not a field in the z struct.")

    return returnedFeatures

def concatenate(zarrays):
    '''
    concatenates zarrays, since np.append/np.concatenate will not work.
    inputs:
        zarrays:    list-like of zarrays to be concatenated
    '''
    ztemp = []
    for i,z in enumerate(zarrays):
        ztemp = ztemp+z.tolist()
    ztemp = zarray(ztemp)
    return ztemp

# NOTE: Certain np functions will not work, like np.copy, as they return ndarrays. Try instead to use functions like
# zarray.copy (the ndarray.copy) which will work.
class zarray(np.ndarray):
    '''
    Almost identical to ndarray of z-trial objects, but allows for additional functionality to be added as methods.
    Should seamlessly integrate such that you can treat ZArrays like any normal ndarray of objects.

    Must be construct by inputting a list or array of z-trial type objects.
    '''
    def __new__(cls, z):
        '''
        Set new attributes in here. This is the de facto init for ndarrays.
        '''
        zarr = np.asarray(z,dtype=object).view(cls)
        #TODO address fieldnames that might differ if two runs from diff models are concatenated.
        zarr._fieldnames = zarr[0]._fieldnames
        return zarr


    def __array_finalize__(self, z):
        '''
        Since ndarrays may be created through other means than instantiation, ex. view-casting or slicing (z2 = z[1:4])
        array finalize can act as the cleanup to make sure that all attributes are carried over. In the case of attrs
        like fieldnames, its as simple as setting the same attribute in this part.
        '''
        if z is None: return
        self._fieldnames = getattr(z, '_fieldnames', None)

    def asdataframe(self):
        """
        Get ZStruct in pd.DataFrame form
        :return: df - The DataFrame version of the z struct
        """
        headers = self.fieldnames
        data = []
        for trial in self:
            thistrial = []
            for attr in headers:
                val = getattr(trial, attr)
                thistrial.append(val)
            data.append(thistrial)
        df = pd.DataFrame(data, columns=headers)
        return df

    '''
    if adding an attribute that might affect the underlying z-trials, like field names, consider using a property
    attribute which can use custom getters and setters that will properly reassign all field names (or keep you from
    changing them)
    '''
    # TODO Add ability to change field names maybe?
    def setfn(self):
        NotImplementedError("Changing Field Names is not supported")
    def getfn(self):
        if self._fieldnames is None: Warning("No field names set, something may have gone wrong. Fields may still be present")
        return self._fieldnames
    def delfn(self):
        NotImplementedError("Deleting Fields is not supported")
    fieldnames = property(getfn, setfn, delfn, "Access field names")

def PrintZStruct(zstruct, printAllTrials=False):
    '''
    First written by JCostello June 2021

    Prints out a preview of a zstruct, similar to Matlab's struct viewer. Use ZStructTranslator to first load a zstruct.
    Multi-dimensional data types are just printed as their size.

    Example usage:
    zstruct = ZStructTranslator.ZStructTranslator(runpath, overwrite=False, use_py=False).get_list()
    PrintZStruct(zstruct)

    Inputs:
        zstruct         - the zstruct object loaded using ZStructTranslator
        printAllTrials  - (optional) If true, prints all trials, otherwise prints a short preview (default)
    '''

    # load into dataframe
    df =  zstruct.asdataframe()

    # print out
    pd.set_option('max_columns', None)
    pd.set_option('display.width', 600)
    if printAllTrials:
        pd.set_option('max_rows', None)
    else:
        pd.reset_option('max_rows')
    print(df)