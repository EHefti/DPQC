classdef dataCompatibilityInterface < handle
    % DATACOMPATIBILITYINTERFACE is an 'Interface Superclass' (see Matlab 
    % help for more info). Every class to extract data from the recordings
    % (as the HDF5 class) should be derived as a subclass of 
    % DATACOMPATIBILITYINTERFACE. This superclass is in chager of checking 
    % that the data extracted from the recording is compatible with the
    % format expected by the 'fileManager'. This data compatibility check 
    % is important to make sure that even when the recording format
    % changes, the high level functions like: 'ActivityMap' or 'AxonTraces'
    % can still be used.
    % 
    % Ideally this scripts shouldn't be changed. But if the user wants 
    % new restrictions or formats to be checked they can anyway be added 
    % here. However, after adding checks here, the user should call them in 
    % 'fileInterface'. 
    %
    % DATACOMPATIBILITYINTERFACE also defines the properties common to all
    % the classes used to extract data from the recordings (as the HDF5 
    % class). These properties are:
    %
    %  info:             - Any aditional information considered important
    %                      from the recording
    %  version:          - Version of the recording
    %  startTime:        - Time when recording started
    %  stopTime:         - Time when recording stoped
    %  samplingFreq:     - Sampling frequency used 
    %  dataLenSamples:   - Recording length in number of samples
    %  dataLenTime:      - Recording length in seconds
    %  firstFrameNum:    - Number (defined internally in the FPGA) that 
    %                      indicates when the recording started
    %  lsb:              - Least significant bit
    %  nChannels:        - Number of channels used for recording
    %  pointerToData:    - Pointer that indicates where the data are stored 
    %                      inside the recording file
    %  map:              - Struct containing 'channel', 'electrode', 'x',
    %                      and 'y' information about the electrodes used for
    %                      recording
    %  spikes:           - Struct containing 'frameno', 'channel', and
    %                      'amplitude' of the spikes detected by the
    %                      recording software
    % 
    % 
    
    properties
        info;
        version;
        startTime;
        stopTime;
        samplingFreq;
        dataLenSamples;
        dataLenTime;
        firstFrameNum;
        lsb;
        gain;
        hpf;
        nChannels;
        pointerToData;
        map;
        spikes;
    end
    
    methods
        
        function checkVersionCompatibility(this)
            
            if ~(ischar(this.version) || isnumeric(this.version) || iscell(this.version))
                errorMsg1 = sprintf('Data compatibility issue: Incorrect "version" format. This could lead to problems later on when running analysis'); 
                errorMsg2 = sprintf('Please make sure "version" is a string, number, or cell');
                error([errorMsg1, newline, errorMsg2]);
            end                           
        end
                
        function checkTimeCompatibility(this)
            if ~(ischar(this.startTime) || ischar(this.stopTime))
                errorMsg1 = sprintf('Data compatibility issue: Incorrect "startTime" or "stopTime" format. This could lead to problems later on when running analysis'); 
                errorMsg2 = sprintf('Please make sure the time is a string');
                error([errorMsg1, newline, errorMsg2]);
            end     
        end
        
        function checkSamplingFreqCompatibility(this)
            if ~(isnumeric(this.samplingFreq) && (this.samplingFreq > 0))
                errorMsg1 = sprintf('Data compatibility issue: Incorrect "samplingFreq" format. This could lead to problems later on when running analysis'); 
                errorMsg2 = sprintf('Please make sure the sampling frequency is a positive integer');
                error([errorMsg1, newline, errorMsg2]);
            end     
        end
        
        function checkLenSamplesCompatibility(this)
            if ~(isnumeric(this.dataLenSamples) && (this.dataLenSamples >= 0))
                errorMsg1 = sprintf('Data compatibility issue: Incorrect "dataLenSamples" format. This could lead to problems later on when running analysis'); 
                errorMsg2 = sprintf('Please make sure the data length in samples is a positive integer');
                error([errorMsg1, newline, errorMsg2]);
            end     
        end
        
        function checkPointerToDataCompatibility(this)
            if ~(ischar(this.pointerToData))
                errorMsg1 = sprintf('Data compatibility issue: Incorrect "pointerToData" format. This could lead to problems later on when running analysis'); 
                errorMsg2 = sprintf('Please make sure the pointer to data is a valid string');
                error([errorMsg1, newline, errorMsg2]);
            end     
        end
        
        function checkFirstFrameNumCompatibility(this)
            if ~(ischar(this.firstFrameNum) || isnumeric(this.firstFrameNum) || iscell(this.firstFrameNum))
                errorMsg1 = sprintf('Data compatibility issue: Incorrect "firstFrameNum" format. This could lead to problems later on when running analysis');
                errorMsg2 = sprintf('Please make sure the first frame number is valid');
                error([errorMsg1, newline, errorMsg2]);
            end
        end
        
        function checkMapCompatibility(this)
            if isstruct(this.map)
                if ~(isfield(this.map, 'channel') && isfield(this.map, 'electrode') && isfield(this.map, 'x') && isfield(this.map, 'y'))
                    errorMsg1 = sprintf('Data compatibility issue: Incorrect "map" format. This could lead to problems later on when running analysis');
                    errorMsg2 = sprintf('Please make sure "map" contains the following fields: "channel", "electrode", "x", and "y"');
                    error([errorMsg1, newline, errorMsg2]);
                end
                
                if any(this.map.channel <= 0)
                    error('Channel numbers should be larger than zero');
                end
                
                if any(this.map.electrode < 0)
                    error('Electrode numbers should be equal or larger than zero');
                end
                
                if any(this.map.x < 0)
                    error('x coordinates should be all positive');
                end
                
                if any(this.map.y < 0)
                    error('y coordinates should be all positive');
                end
                
            else
                errorMsg1 = sprintf('Data compatibility issue: Incorrect "map" format. This could lead to problems later on when running analysis');
                errorMsg2 = sprintf('Please make sure the map is a valid struct');
                error([errorMsg1, newline, errorMsg2]);
            end
        end
        
        function checkSpikesCompatibility(this)
            if isstruct(this.spikes)
                if ~(isfield(this.spikes, 'frameno') && isfield(this.spikes, 'channel') && isfield(this.spikes, 'amplitude'))
                    errorMsg1 = sprintf('Data compatibility issue: Incorrect "spikes" format. This could lead to problems later on when running analysis');
                    errorMsg2 = sprintf('Please make sure "spikes" contains the following fields: "frameno", "channel", and "amplitude"');
                    error([errorMsg1, newline, errorMsg2]);
                end
                
                if any(this.spikes.channel <= 0)
                    error('Channel numbers should be larger than zero');
                end
                
            else
                errorMsg1 = sprintf('Data compatibility issue: Incorrect "spikes" format. This could lead to problems later on when running analysis');
                errorMsg2 = sprintf('Please make sure the spikes is a valid struct');
                error([errorMsg1, newline, errorMsg2]);
            end
        end
        
        function checkLsbCompatibility(this)
            if ~(isnumeric(this.lsb) && (this.lsb > 0))
                errorMsg1 = sprintf('Data compatibility issue: Incorrect "lsb" format. This could lead to problems later on when running analysis'); 
                errorMsg2 = sprintf('Please make sure the lsb is a positive number');
                error([errorMsg1, newline, errorMsg2]);
            end     
        end
        
        function checkGainCompatibility(this)
            if ~(isnumeric(this.gain) && (this.gain > 0))
                errorMsg1 = sprintf('Data compatibility issue: Incorrect "gain" format. This could lead to problems later on when running analysis'); 
                errorMsg2 = sprintf('Please make sure the gain is a positive number');
                error([errorMsg1, newline, errorMsg2]);
            end     
        end
        
        function checkHpfCompatibility(this)
            if ~(isnumeric(this.hpf) && (this.hpf > 0))
                errorMsg1 = sprintf('Data compatibility issue: Incorrect "hpf" format. This could lead to problems later on when running analysis'); 
                errorMsg2 = sprintf('Please make sure the hpf is a positive number');
                error([errorMsg1, newline, errorMsg2]);
            end     
        end

        function checkNChannelsCompatibility(this)
            if ~(isnumeric(this.nChannels) && (this.nChannels > 0))
                errorMsg1 = sprintf('Data compatibility issue: Incorrect "nChannels" format. This could lead to problems later on when running analysis'); 
                errorMsg2 = sprintf('Please make sure the number of channels is a positive number');
                error([errorMsg1, newline, errorMsg2]);
            end     
        end
    end
end

