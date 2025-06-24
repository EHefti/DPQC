classdef HDF5_2 < mxw.dataFunctions.dataCompatibilityInterface_2
    % HDF5_2 is a the class that directly interacts with the recording file
    % extracting its information. It handles hdf5 files of the new format.
    % The class is a subclass of 'dataCompatibilityInterface_2', and contains
    % all its properties. The methods of this class are called from the
    % 'fileInterface2' class. To add any other class that handles different
    % file formats the user should follow the structure of the HDF5 class.
    %
    %
    
    properties
    end
    
    methods
        
        function gobj = HDF5_2()
        end
        
        function extractDataTime(this, fileObj)
            
            defaultValue = false;
            
            try
                start_epoch= h5read(fileObj.filePath,['/wells/' fileObj.wellID '/' fileObj.recID '/start_time']);
                stop_epoch= h5read(fileObj.filePath,['/wells/' fileObj.wellID '/' fileObj.recID '/stop_time']);
                this.startTime = datestr(datetime(start_epoch/1000, 'convertfrom','posixtime','TimeZone','local','Format','yyyy-MM-dd HH:mm:ss'));
                this.stopTime = datestr(datetime(stop_epoch/1000, 'convertfrom','posixtime','TimeZone','local','Format','yyyy-MM-dd HH:mm:ss'));
                this.dataLenTime = double((stop_epoch - start_epoch)/(1000));
                defaultValue = true;
            catch
                this.startTime = 'no time extracted';
                this.stopTime = 'no time extracted';
                this.dataLenTime = 'no time extracted';
            end
            
            if defaultValue
                this.checkTimeCompatibility();
                this.checkLenTimeCompatibility();
            end
        end
        
        function extractDataSamplingFreqLenSamples(this, fileObj)
            
            defaultValue = false;
            
            try           
                this.samplingFreq = h5read(fileObj.filePath,['/wells/' fileObj.wellID '/' fileObj.recID '/settings/sampling']);
                this.dataLenSamples = double(this.dataLenTime*this.samplingFreq);
                defaultValue = true;
            catch
                this.samplingFreq = 'no sampling frequency extracted';
                this.dataLenSamples = 'no data length in samples extracted';
            end
            
            if defaultValue
                this.checkSamplingFreqCompatibility();
                this.checkLenSamplesCompatibility();
            end          
        end
        
        function defineDataPointerToData(this, fileObj)
            
            defaultValue = false;
            
            try
                this.pointerToData = ['/wells/' fileObj.wellID '/' fileObj.recID '/groups/routed/raw/'];
                h5info(fileObj.filePath, this.pointerToData);
                defaultValue = true;
            catch
                this.pointerToData = 'no pointer extracted';
            end
            
            if defaultValue
                this.checkPointerToDataCompatibility();
            end
        end
        
        function extractDataLsb(this, fileObj)
            
            defaultValue = false;
            
            try
                % load lsb in [V]
                lsb = h5read(fileObj.filePath,['/wells/' fileObj.wellID '/' fileObj.recID '/settings/lsb']);

                % store lsb in [uV]
                this.lsb = lsb*1e6;
                
                defaultValue = true;
            catch
                this.lsb = 'no lsb extracted';
            end
            
            if defaultValue
                this.checkLsbCompatibility();
            end   
        end
        
        function extractDataGain(this, fileObj)
            
            defaultValue = false;
            
            try
                this.gain = h5read(fileObj.filePath,['/wells/' fileObj.wellID '/' fileObj.recID '/settings/gain']);
                defaultValue = true;
            catch
                this.gain = 'no gain extracted';
            end
            
            if defaultValue
                this.checkGainCompatibility();
            end               
        end        
        
        function extractDataHpf(this, fileObj)
            
            defaultValue = false;
            
            try
                this.hpf = h5read(fileObj.filePath,['/wells/' fileObj.wellID '/' fileObj.recID '/settings/hpf']);
                defaultValue = true;
            catch
                this.hpf = 'no hpf extracted';
            end
            
            if defaultValue
                this.checkHpfCompatibility();
            end
        end
        
        function extractDataTemperature(this, fileObj)
            
            defaultValue = false;
            
            try
                this.temperature = h5read(fileObj.filePath,'/environment/temperature');
                timeVector = datetime(this.temperature.Time, 'convertfrom','posixtime','TimeZone','local','Format','yyyy-MM-dd HH:mm:ss');
                this.temperature.Time = datestr(timeVector);
                this.temperature.Seconds = seconds(timeVector-timeVector(1,1));
                defaultValue = true;
            catch
                this.temperature = 'no temperature extracted';
            end
            
            if defaultValue
                this.checkTemperatureCompatibility();
            end               
        end
        
        function extractDataSpikeThreshold(this, fileObj)
            
            defaultValue = false;
            
            try
                this.spikeThreshold = h5read(fileObj.filePath,['/wells/' fileObj.wellID '/' fileObj.recID '/settings/spike_threshold']);
                defaultValue = true;
            catch
                this.spikeThreshold = 'no spike threshold extracted';
            end
            
            if defaultValue
                this.checkSpikeThresholdCompatibility();
            end
        end
        
        function extractDataMapNChannels(this, fileObj)
            
            defaultValue = false;
           
            try
                map = h5read(fileObj.filePath,['/wells/' fileObj.wellID '/' fileObj.recID '/settings/mapping']);
                
                % check and correct for duplicate values in the mapping (SW-228):
                
                A = [map.channel map.electrode map.x map.y];
                [~, inds] = unique(A,'rows');
                
                this.nChannels = length(inds);
                
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
                
                defaultValue = true;
            catch
                this.nChannels = 'no number of channels extracted';
                this.map = 'no map extracted';
            end
            
            if defaultValue
                this.checkNChannelsCompatibility();
                this.checkMapCompatibility();
            end
        end
        
        function extractDataSpikes(this, fileObj)

            defaultValue = false;
            
            try
                this.spikes = h5read(fileObj.filePath,['/wells/' fileObj.wellID '/' fileObj.recID '/spikes']);
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
                defaultValue = true;
            catch
                this.spikes = 'no spikes extracted';
            end
            
            if defaultValue
                this.checkSpikesCompatibility();
            end
        end
                
        function extractFirstFrameNum(this, fileObj)
            
            defaultValue = false;
            
            % if raw data recorded
            if ~strcmp(this.pointerToData,'no pointer extracted')
                try
                    frame_nos = h5read(fileObj.filePath,['/wells/' fileObj.wellID '/' fileObj.recID '/groups/routed/frame_nos/']);
                    this.firstFrameNum = double(frame_nos(1));
                catch
                    this.firstFrameNum = 'no first frame number extracted';
                end    
            else
                if isnumeric(this.spikes.frameno)
                    try
                        this.firstFrameNum = double(min(this.spikes.frameno)-1);
                        defaultValue = true;
                    catch
                        this.firstFrameNum = 'no first frame number extracted';
                    end
                end  
            end
            
            if defaultValue
                this.checkFirstFrameNumCompatibility();
            end
        end
        
        function data = extractDataFullRawData(this, fileObj, start, len)
            
            data = double(h5read(fileObj.filePath, fileObj.pointerToData, [start 1], [len, fileObj.nChannels])) * fileObj.lsb;

        end
        
        function data = extractDataRawData(this, fileObj, start, len, electrodes)
            
            index = mod(find(fileObj.map.electrode == electrodes), length(fileObj.map.electrode));
            
            if isscalar(index)
                data = double(h5read(fileObj.filePath, fileObj.pointerToData, [start index], [len, 1])) * fileObj.lsb;
                
            elseif isvector(index)
                data = zeros(len, length(index));
                
                for i = 1:length(index)
                    data(:,i) = double(h5read(fileObj.filePath, fileObj.pointerToData, [start index(i)], [len, 1])) * fileObj.lsb;
                end
            end

        end
        
%         function data = extractDataDAC(this, fileObj, start, len)
%             data = double(h5read(fileObj.filePath, fileObj.pointerToData, [start fileObj.nChannels+1], [len, 1]));
%         end
                
    end
end
