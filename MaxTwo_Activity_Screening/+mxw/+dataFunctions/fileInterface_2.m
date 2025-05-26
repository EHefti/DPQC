classdef fileInterface_2 < handle
    % The FILEINTERFACE class is in charge of calling the functions that 
    % direclty extract the information from the recordings of the new 
    % format (such as the funcions of the HDF5 class), and pass the output 
    % of those functions by the compatibility functions defined in 
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
        wellID;
        recID;
        extension;
        dataFormat;
        startTime;
        stopTime;
        samplingFreq;
        dataLenSamples;
        dataLenTime;
        firstFrameNum;
        lsb;
        gain;
        hpf;        
        temperature;
        spikeThreshold; 
        nChannels;
        pointerToData;
        map;
        spikes;
        dynamicDataStorge;
    end
    
    methods
        
        function this = fileInterface_2(filePath,well_label,rec_label)
            
            if nargin > 0
                this.filePath = filePath;
                this.wellID = well_label;
                this.recID = rec_label;
                this.extractExtension();                
                this.defineDataFormat;
                this.extractTime();
                this.extractSamplingFreqLenSamples();
                this.definePointerToData();
                this.extractLsb();
                this.extractGain();
                this.extractHpf();
                this.extractTemperature();
                this.extractSpikeThreshold();
                this.extractMapNChannels();
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
                    this.dataFormat = mxw.dataFunctions.HDF5_2();
                otherwise
                    error('This data format can not be handled')
            end
        end
        
        function extractTime(this)
            
            this.dataFormat.extractDataTime(this);
            
            this.startTime = this.dataFormat.startTime;
            this.stopTime = this.dataFormat.stopTime;
            this.dataLenTime = this.dataFormat.dataLenTime;
        end
        
        function extractSamplingFreqLenSamples(this)
            
            this.dataFormat.extractDataSamplingFreqLenSamples(this);

            this.samplingFreq = this.dataFormat.samplingFreq;
            this.dataLenSamples = this.dataFormat.dataLenSamples;
        end
        
        function definePointerToData(this)
            
            this.dataFormat.defineDataPointerToData(this);
            
            this.pointerToData = this.dataFormat.pointerToData;
        end

        function extractLsb(this)

            this.dataFormat.extractDataLsb(this);
            
            this.lsb = this.dataFormat.lsb;
        end
        
        function extractGain(this)
            
            this.dataFormat.extractDataGain(this);
                  
            this.gain = this.dataFormat.gain;
        end
        
        function extractHpf(this)
            
            this.dataFormat.extractDataHpf(this);
                   
            this.hpf = this.dataFormat.hpf;
        end

        function extractTemperature(this)
            
            this.dataFormat.extractDataTemperature(this);
                   
            this.temperature = this.dataFormat.temperature;
        end        
        
        function extractSpikeThreshold(this)
            
            this.dataFormat.extractDataSpikeThreshold(this);

            this.spikeThreshold = this.dataFormat.spikeThreshold;
        end                
        
        function extractMapNChannels(this)
            
            this.dataFormat.extractDataMapNChannels(this);
          
            this.map = this.dataFormat.map;
            this.nChannels = this.dataFormat.nChannels;
        end
        
        function extractSpikes(this)
            
            this.dataFormat.extractDataSpikes(this);

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
       
    end
end

