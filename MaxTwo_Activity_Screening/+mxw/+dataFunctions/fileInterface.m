classdef fileInterface < handle
    % The FILEINTERFACE class is in charge of calling the functions that 
    % direclty extract the information from the recordings (such as the 
    % funcions of the HDF5 class), and pass the output of those functions 
    % by the compatibility functions defined in 
    % 'dataCompatibilityInterface'. Therefore this class is the interface
    % between the recording files and the 'fileManager' that handles all
    % the analysis.
    %
    % This class shouldn't be instantiated by the user since it is
    % automatically instantiated when the 'fileManager' object is created.
    %
    % This class shares most of the properties of the classes that direclty
    % interact with the recordings (such as the HDF5 class), and has some
    % extra ones:
    %   
    %
    %  filePath:            - 'path/to/data'
    %  extension:           - Extension of the recording (e.g. '.raw.h5')
    %  dataFormat:          - Object from the class defined in 'extension'.
    %                         This object is key to define the functions    
    %                         that extract the information from the 
    %                         recordings
    %  dynamicDataStorage:  - Array that works as a cache memory containing
    %                         data from the recordings under analysis
    %
    % To handle new data formats the following steps have to be followed:
    %   1- In the function 'extractExtension' extract the characters from
    %   'filePath' that contain the format extension
    %   2- Create a case with the extension
    %   3- In the function 'defineDataFormat' create a case with the
    %   corresponding extension, and inside that case instantiate the class
    %   that handles the format
    %   4- Create the class that handles the new data format as a subclass
    %   of 'dataCompatibilityInterface' (like the HDF5 class)
    %
    %
    
    properties
        filePath;
        extension;
        dataFormat;
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
        dynamicDataStorge;
    end
    
    methods
        
        function this = fileInterface(filePath)
            
            if nargin > 0
                this.filePath = filePath;                
                this.extractExtension();                
                this.defineDataFormat;
                this.extractVersion();
                this.extractTime();
                this.extractSamplingFreq();
                this.definePointerToData();
                this.extractLenSamples();
                this.extractLsb();
                this.extractGain();
                this.extractHpf();
                this.extractNChannels();
                this.extractMap();
                this.extractSpikes();
                this.extractFirstFrameNum();                
            end
        end
        
        function extractExtension(this)
            
            extension = this.filePath(end-6:end);
            
            switch extension
                case '.raw.h5'
                    this.extension = extension;
                otherwise
                    error('Folders contains not only recording files')
            end    
        end
        
        function defineDataFormat(this)
            
            switch this.extension
                case '.raw.h5'
                    this.dataFormat = mxw.dataFunctions.HDF5();
                otherwise
                    error('This data format can not be handled')
            end
        end
        
        function extractVersion(this)
            
            defaultValue = false;
            
            try
                this.dataFormat.extractDataVersion(this);
            catch
                defaultValue = true;
            end
            
            if defaultValue
                this.dataFormat.version = 'no version extracted';
            else
                this.dataFormat.checkVersionCompatibility();
            end
            
            this.version = this.dataFormat.version;
        end
        
        function extractTime(this)
            
            defaultValue = false;
            
            try
                this.dataFormat.extractDataTime(this);
            catch
                defaultValue = true;
            end
            
            if defaultValue
                this.dataFormat.startTime = 'no time extracted';
                this.dataFormat.stopTime = 'no time extracted';
            else
                this.dataFormat.checkTimeCompatibility();
            end
            
            this.startTime = this.dataFormat.startTime;
            this.stopTime = this.dataFormat.stopTime;
        end
        
        function extractSamplingFreq(this)
            
            defaultValue = false;
            
            try
                this.dataFormat.extractDataSamplingFreq(this);
            catch
                defaultValue = true;
            end
            
            if defaultValue
                this.dataFormat.samplingFreq = 'no sampling frequency extracted';
            else
                this.dataFormat.checkSamplingFreqCompatibility();
            end
            
            this.samplingFreq = this.dataFormat.samplingFreq;
        end
        
        function extractLenSamples(this)
            
            defaultValue = false;
            
            try
                this.dataFormat.extractDataLenSamples(this);
            catch
                defaultValue = true;
            end
            
            if defaultValue
                this.dataFormat.dataLenSamples = 'no length extracted';
            else
                this.dataFormat.checkLenSamplesCompatibility();
            end
            
            this.dataLenSamples = this.dataFormat.dataLenSamples;
        end
        
        function definePointerToData(this)
            
            defaultValue = false;
            
            try
                this.dataFormat.defineDataPointerToData(this);
            catch
                defaultValue = true;
            end
            
            if defaultValue
                this.dataFormat.pointerToData = 'no pointer extracted';
            else
                this.dataFormat.checkPointerToDataCompatibility();
            end
            
            this.pointerToData = this.dataFormat.pointerToData;
        end
        
        function extractMap(this)
            
            defaultValue = false;
            
            try
                this.dataFormat.extractDataMap(this);
            catch
                defaultValue = true;
            end
            
            if defaultValue
                this.dataFormat.map = 'no map extracted';
            else
                this.dataFormat.checkMapCompatibility();
            end
            
            this.map = this.dataFormat.map;
        end
        
        function extractSpikes(this)
            
            defaultValue = false;
            
            try
            this.dataFormat.extractDataSpikes(this);
            catch
                defaultValue = true;
            end
            
            if defaultValue
                this.dataFormat.spikes = 'no spikes extracted';
            else
                this.dataFormat.checkSpikesCompatibility();
            end
            
            this.spikes = this.dataFormat.spikes;
        end
                
        function extractFirstFrameNum(this)
            
            defaultValue = false;
            
            try
                this.dataFormat.extractFirstFrameNum(this);
            catch
                defaultValue = true;
            end
            
            if defaultValue
                this.dataFormat.firstFrameNum = 'no first frame number extracted';
            else
                this.dataFormat.checkFirstFrameNumCompatibility();
            end
            
            this.firstFrameNum = this.dataFormat.firstFrameNum;
        end        
        
        function data = extractFullRawData(this, start, len)
            
            data = this.dataFormat.extractDataFullRawData(this, start, len);
        end
        
        function data = extractRawData(this, start, len, electrodes)
            
            data = this.dataFormat.extractDataRawData(this, start, len, electrodes);
        end
        
        function data = extractDAC(this, start, len)
            
            data = this.dataFormat.extractDataDAC(this, start, len);
        end
        
        function extractLsb(this)
            
            defaultValue = false;
            
            try
                this.dataFormat.extractDataLsb(this);
            catch
                defaultValue = true;
            end
            
            if defaultValue
                
                % old files do not have the lsb saved. In that case,
                % extract the gain here and derive the lsb from the gain.
                
                tmp = this;
                tmp.dataFormat.extractDataGain(tmp);
                tmp_gain = tmp.dataFormat.gain;
                
                if tmp_gain == 512
                    lsb = 6.2942;
%                 elseif gain = 1024
%                     lsb = 
                else 
                    lsb = 'no lsb extracted';
                end 
                
                this.dataFormat.lsb = lsb;
            else
                this.dataFormat.checkLsbCompatibility();
            end
            
            this.lsb = this.dataFormat.lsb;
        end
        
        function extractGain(this)
            
            defaultValue = false;
            
            try
                this.dataFormat.extractDataGain(this);
            catch
                defaultValue = true;
            end
            
            if defaultValue
                this.dataFormat.gain = 'no gain extracted';
            else
                this.dataFormat.checkGainCompatibility();
            end
            
            this.gain = this.dataFormat.gain;
        end
        
        function extractHpf(this)
            
            defaultValue = false;
            
            try
                this.dataFormat.extractDataHpf(this);
            catch
                defaultValue = true;
            end
            
            if defaultValue
                this.dataFormat.hpf = 'no hpf extracted';
            else
                this.dataFormat.checkHpfCompatibility();
            end
            
            this.hpf = this.dataFormat.hpf;
        end
        
        function extractNChannels(this)
            
            defaultValue = false;
            
            try
                this.dataFormat.extractDataNChannels(this);
            catch
                defaultValue = true;
            end
            
            if defaultValue
                this.dataFormat.nChannels = 'no number of channels extracted';
            else
                this.dataFormat.checkNChannelsCompatibility();
            end
            
            this.nChannels = this.dataFormat.nChannels;
        end
    end
end

