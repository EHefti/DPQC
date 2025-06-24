classdef HDF5 < mxw.dataFunctions.dataCompatibilityInterface
    % HDF5 is a the class that directly interacts with the recording file
    % extracting its information. It can only handles hdf5 format files.
    % The class is a subclass of 'dataCompatibilityInterface', and contains
    % all its properties. The methods of this class are called from the
    % 'fileInterface' class. To add any other class that handles different
    % file formats the user should follow ths structure of the HDF5 class.
    %
    %
    
    properties
    end
    
    methods
        
        function gobj = HDF5()
        end
        
        function extractDataVersion(this, fileObj)
            
            try
                this.version = h5read(fileObj.filePath, '/version');
            catch
                this.version = cellstr('no version information');
            end
        end
        
        function extractDataTime(this, fileObj)
            
            timeStr = h5read(fileObj.filePath, '/time');
            
            startTime = extractAfter(timeStr{1},'start: ');
            this.startTime = extractBefore(startTime,20);
            
            stopTime = extractAfter(timeStr{1},'stop: ');
            this.stopTime = extractBefore(stopTime,20);
            
        end
        
        function extractDataSamplingFreq(this, fileObj)
            
            % disp('Data sampling frequency to be defined');
            this.samplingFreq = 20000;
        end
        
        function extractDataLenSamples(this, fileObj)
            
            dataInfo = h5info(fileObj.filePath, fileObj.pointerToData);
            this.dataLenSamples = dataInfo.Dataspace.Size(1);
            
            % 05.09.2018: Workaround for Spike-only files
            
            if this.dataLenSamples == 0
                
                startTime = datetime(this.startTime);
                stopTime = datetime(this.stopTime);
                
                duration = seconds(stopTime-startTime);
                this.dataLenSamples = duration*this.samplingFreq;
                
            end
        end
        
        function defineDataPointerToData(this, fileObj)
            
            this.pointerToData = '/sig';
        end
        
        function extractDataLsb(this, fileObj)
            
            % load lsb in [V]
            lsb = h5read(fileObj.filePath, '/settings/lsb');
            % store lsb in [uV]
            this.lsb = lsb*1e6;
        end
        
        function extractDataGain(this, fileObj)
            
            this.gain = h5read(fileObj.filePath, '/settings/gain');
        end
        
        function extractDataHpf(this, fileObj)
            
            this.hpf = h5read(fileObj.filePath, '/settings/hpf');
        end
        
        function extractDataNChannels(this, fileObj)
            
            %             disp('Number of channels to be defined');
            this.nChannels = 1024;
        end
        
        function extractDataMap(this, fileObj)
            
            mapFromFile = h5read(fileObj.filePath, '/mapping');
            
            
            % check and correct for duplicate values in the mapping (SW-228):
            map = mapFromFile;
            
            A = [map.channel map.electrode map.x map.y];
            [~, inds] = unique(A,'rows');
            
            if length(inds)<size(A,1) % duplicate 
                map.channel = map.channel(inds);
                map.electrode = map.electrode(inds);
                map.x = map.x(inds);
                map.y = map.y(inds);
            end
            
            map.channel = map.channel(map.electrode >= 0);
            map.channel = map.channel + int32(ones(size(map.channel)));
            map.x = map.x(map.electrode >= 0);
            map.y = map.y(map.electrode >= 0);
            map.electrode = map.electrode(map.electrode >= 0);
            
            % removed in order to have same el-number on scope and matlab:
            % map.electrode = map.electrode + int32(ones(size(map.electrode)));
            
            this.map = map;
        end
        
        function extractDataSpikes(this, fileObj)
            
            this.spikes = h5read(fileObj.filePath, '/proc0/spikeTimes');
            this.spikes.channel = this.spikes.channel + int32(ones(size(this.spikes.channel)));
            this.spikes.amplitude = this.spikes.amplitude * fileObj.lsb;
            
            spikesFromRoutedChannels = true(length(this.spikes.channel), 1);
            
            for i = 1:length(this.spikes.channel)
                if ~any(this.spikes.channel(i) == this.map.channel)
                    spikesFromRoutedChannels(i) = false;
                end
            end
             
            this.spikes.frameno = this.spikes.frameno(spikesFromRoutedChannels);
            this.spikes.channel = this.spikes.channel(spikesFromRoutedChannels);
            this.spikes.amplitude = this.spikes.amplitude(spikesFromRoutedChannels);
        end
        
        function extractFirstFrameNum(this, fileObj)
            
            frameNumInfo = h5info(fileObj.filePath, fileObj.pointerToData);
            if frameNumInfo.Dataspace.Size(1)>0
                rawFrameNum = h5read(fileObj.filePath, fileObj.pointerToData, [1 1027], [1, 2]);
                this.firstFrameNum = bitor(bitshift(double(rawFrameNum(:,2)), 16), double(rawFrameNum(:,1)));
            else
                this.firstFrameNum = double(min(this.spikes.frameno)-1);
            end
        end        
        
        function data = extractDataFullRawData(this, fileObj, start, len)
            
            data = double(h5read(fileObj.filePath, fileObj.pointerToData, [start 1], [len, fileObj.nChannels])) * fileObj.lsb;
            data = data(:,fileObj.map.channel);
        end
        
        function data = extractDataRawData(this, fileObj, start, len, electrodes)
            
            index = mod(find(fileObj.map.electrode == electrodes), length(fileObj.map.electrode));
            index(index == 0) = length(fileObj.map.electrode);
            channels = double(fileObj.map.channel(index));
            
            if isscalar(channels)
                channelsData = double(h5read(fileObj.filePath, fileObj.pointerToData, [start channels], [len, 1])) * fileObj.lsb;
                
            elseif isvector(channels)
                channelsData = zeros(len, length(channels));
                
                for i = 1:length(channels)
                    channelsData(:,i) = double(h5read(fileObj.filePath, fileObj.pointerToData, [start channels(i)], [len, 1])) * fileObj.lsb;
                end
            end
            
            data = channelsData;
        end
        
        function data = extractDataDAC(this, fileObj, start, len)
            data = double(h5read(fileObj.filePath, fileObj.pointerToData, [start fileObj.nChannels+1], [len, 1]));
        end
        
    end
end
