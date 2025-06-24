classdef fileManager < handle
    % The FILEMANAGER is the main class for any kind of analysis. It has to
    % be instatiated in order to start working with the Matlab Toolbox. To
    % instantiate it use the path to the data as an input.
    %
    % recObj = mxw.fileManager('path/to/data')
    %
    % recObj is an object with the following properties:
    %
    %  referencePath:    - 'path/to/data'
    %  wellNo:           - Number of well (1-6) from which to analyse
    %                      recording (for MaxTwo data. For MaxOne, default
    %                      value of 1 is used)
    %  isFolder:         - Boolean:(1 if 'path/to/data' points to a folder,
    %                      0 if it does not)
    %  nFiles:           - Number of files in 'path/to/data'. Always = 1
    %                      for the new format.
    %  nRecordings:      - Number of recordings (equal to number of files
    %                      in Legacy format and variable in the new format.
    %  fileNameList:     - Name of files in 'path/to/data'
    %  fileObj:          - Array of objects instantiated from class
    %                      FILEINTERFACE, each object contains the
    %                      informatoin of each individual file in
    %                      'path/to/data'
    %  rawMap:           - Array of structures containing the map and
    %                      spikes extracted from each individual file
    %                      listed on the 'fileObj' property
    %  processedMap:     - Structure containing the 'x' and 'y' positions
    %                      and the electrode number of all the electrodes
    %                      used among all the recordings listed on the
    %                      'fileObj' property. The correspondance between
    %                      electrodes and recordings can be found in the
    %                      field 'recordingIndex'. The non routed
    %                      electrodes among all the recordings are in the
    %                      field 'nonRoutedElec'
    %  extractedSpikes:  - Array of structures containing the 'frameno'
    %                      and 'amplitude' of spikes found by the recording
    %                      unit. Each structure contains the information of
    %                      every recording over the full MEA
    %  bandPassFilter:   - 'bandpass' object containing the information of
    %                      the bandpass filter used every time the
    %                      'recObj.extractBPFData'method is implemented
    %
    % recObj is an object with the following public methods:
    %
    % extractRawData:
    %
    %    to call the method: [] = mxw.fileManager.extractRawData()
    %
    %    -This method extracts the raw data (not bandpass filtered) from
    %     the files listed on 'fileObj'
    %
    %     -The input parameters for this method are:
    %      -start: data sample to start extracting the data
    %      -len: length of the chunk of data to extract (in samples)
    %      -varargin: ...
    %      -'files': file or files to extract from the 'fileObj' array.
    %                File numbers are input as an array [file_1,...,file_n]
    %                corresponding to the 'fileObj" array. If the recObj
    %                object contains only one file, 'files' can be omitted
    %      -'electrodes': electrodes to be extracted. The electrode numbers
    %                     are input as an array: [el_1,...,el_n]
    %
    %     -The output parameters for this method are:
    %      -data: extracted data. The data are sorted in the following
    %             way:
    %              -every row represents a data point
    %              -every column represents an electrode in
    %              'electrodesArray'. Cells of electrodes in
    %              'electrodesArray' at the same time correspond to a file
    %              in 'filesArray'
    %      -filesArray: array of files from which the data were extracted
    %      -electrodesArray: array of cells containing the electrodes from
    %                        which the data were extracted. Every cell
    %                        corresponds to a file in 'filesArray'
    %
    %    -Examples
    %       -Considering recObj contains only one file, and we want to
    %       extract a chunk of data between sample 200 and sample 10000
    %       from all the electrodes:
    %
    %       [data, filesArray, electrodesArray] = extractRawData(200, 9800)
    %
    %       -Same example as before, but now extracting some specific
    %       electrodes:
    %
    %       [data, filesArray, electrodesArray] = extractRawData(200,...
    %           9800, 'electrodes', [230, 740, 3784, 5670, 10081])
    %
    %       -Considering recObj contains several files, and we want to
    %       extract a chunk of data between sample 200 and sample 10000
    %       from a all the electrodes in file number 8:
    %
    %       [data, filesArray, electrodesArray] = extractRawData(200, ...
    %           9800, 'file', 8)
    %
    %       -Considering recObj contains several files, and we want to
    %       extract a chunk of data between sample 200 and sample 10000
    %       from specific electrodes:
    %
    %       [data, filesArray, electrodesArray] = extractRawData(200,...
    %           9800, 'electrodes', [283, 1704, 3478, 5632, 12088])
    %
    % extractCutOuts:
    %
    %    to call the method: [] = mxw.fileManager.extractCutOuts()
    %
    %    -This method extracts cutouts of bandpass filtered data from all
    %     the electrodes in a file
    %
    %     -The input parameters for this method are:
    %      -spikesTimePoints: data samples that define the points where the
    %                         cutouts are extracted
    %      -prePointsSpike: samples that define the cutout to the left of
    %                       'spikesTimePoints'
    %      -postPointsSpike: samples that define the cutout to the right of
    %                        'spikesTimePoints'
    %      -varargin: ...
    %      -'files': file from which the coutouts are extracted. The file
    %                number is referenced to the 'fileObj' array. When
    %                there is just one file, 'files' can be omitted
    %
    %     -The output parameters for this method are:
    %      -waveformCutOuts: extracted cutouts from all the electrodes in
    %                        the chosen file.The cutouts are sorted in the
    %                        following way:
    %                        -each column represents all the extracted
    %                         cutouts for each 'spikesTimePoints'. Each
    %                         column contains the cutouts concatenated in a
    %                         way that the first cutout comes from the
    %                         first electrode in 'electrodesArray' and so
    %                         on. The cutouts are
    %                         'prePointsSpike'+'postPointsSpike' samples
    %                         long, and each column has a length of
    %                         ('prePointsSpike'+'postPointsSpike') times
    %                         the number of electrodes in the file
    %      -electrodesArray: number of electrodes from which the cutouts
    %                        were extracted
    %
    %    -Examples
    %       -Considering recObj contains only one file, and we want to
    %       extract some cutcouts around the data samples of interest:
    %       2450, 5040, and 7900. We want these cutouts to start 30 samples
    %       before the point of interest and end 50 samples after it:
    %
    %       [waveformCutOuts, electrodesArray] = extractCutOuts([2450, ...
    %           5040, 7900], 30, 50)
    %
    %       -Considering recObj contains several files, and we want to
    %       extract the same cutouts as before, but from file number 14:
    %
    %       [waveformCutOuts, electrodesArray] = extractCutOuts([2450, ...
    %           5040, 7900], 30, 50, 'files', 14)
    %
    % extractBPFData:
    %
    %    to call the method: [] = mxw.fileManager.extractBPFData()
    %
    %    -This method extracts the bandpass filtered data from the files
    %     listed on 'fileObj'. It works exaclty as the method
    %     'extractRawData' explained above
    %
    % modifyBPFilter:
    %
    %    to call the method: mxw.fileManager.modifyBPFilter()
    %
    %    -This method modifies the bandpass filter stored in the
    %     'bandPassFilter' property of recObj
    %
    %     -The input parameters for this method are:
    %      -lowCut: frequency cut for high-pass component
    %      -highCut: frequency cut for low-pass component
    %      -order: order of the filter
    %
    %    -Examples
    %       -Considering we want to change the bandpass filter stored in
    %       recObj for the next time we extract the bandpass filtered
    %       data. (Notice that this change would be permanent as far as the
    %       object exists, if, however, another fileManager is instantiated
    %       the bandpass filter will be the default one):
    %
    %       modifyBPFilter(500, 5000, 6)
    %
    %
    
    properties
        referencePath;
        wellID;
        isFolder;
        nFiles;
        nRecordings;
        version;
        fileNameList;
        fileObj;
        rawMap;
        processedMap;
        extractedSpikes;
        bandPassFilter;
    end
    
    methods (Hidden = true)
        
        function this = fileManager(referencePath, wellID)
            
            if nargin > 0
                
                if nargin == 1
                    this.wellID = 1;
                else
                    this.wellID = wellID;
                end
                
                this.referencePath = referencePath;
                this.determineIfFolder();
                this.calculateNumFiles();
                this.createFileNameList();
                this.checkVersion();
                this.createFileObj();
                this.compileRawMaps();
                this.computeMap();
                this.compileSpikes();
                this.cleanNonRoutedElec();
                [lowCut, highCut, order] = this.defaultBPF();
                this.modifyBPFilter(lowCut, highCut, order);
            end
        end
    end
    
    methods (Access = private)
        
        function determineIfFolder(this)
            
            if isdir(this.referencePath)
                this.isFolder = true;
                
            else
                this.isFolder = false;
            end
        end
        
        function calculateNumFiles(this)
            
            if this.isFolder
                info = dir(this.referencePath);
                info = info(3:end,:);
                c=0;
                for i=1:length(info)
                    if length(info(i).name)>6
                        if strcmp(info(i).name(end-6:end),'.raw.h5')
                            c=c+1;
                        end
                    end
                end
                this.nFiles = c;
                
            else
                this.nFiles = 1;
            end
        end
        
        function createFileNameList(this)
            
            if this.isFolder
                filesList = cell(this.nFiles,1);
                info = dir(this.referencePath);
                info = info(3:end,:);
                
                % only consider .raw.h5-files and ignore others
                c=1;
                for iFile=1:length(info)
                    if length(info(iFile).name)>6
                        if strcmp(info(iFile).name(end-6:end),'.raw.h5')
                            filesList(c)= cellstr([this.referencePath, filesep, info(iFile).name]);
                            c=c+1;
                        end
                    end
                end
                
                this.fileNameList = filesList;
                
            else
                this.fileNameList = cellstr(this.referencePath);
            end
        end
        
        function checkVersion(this)
            
            this.version = char(h5read(this.fileNameList{1},'/version'));
        end
        
        function createFileObj(this)
            
            if this.version == '20160704'  % If HDF format Legacy
                
                fileObjects(this.nFiles,1) = mxw.dataFunctions.fileInterface();
                
                for iFile = 1:this.nFiles
                    fileObjects(iFile) = mxw.dataFunctions.fileInterface(char(this.fileNameList(iFile)));
                end
                
                this.nRecordings = this.nFiles;
                
            elseif this.version == '20190530' % IF HDF format New (introduced with MaxTwo)
                
                well_labels = h5info(this.fileNameList{:},'/wells');
                well_labels = {well_labels.Groups.Name};
                well_label = well_labels{this.wellID}(end-6:end);
                
                recordings = h5info(this.fileNameList{:},strcat('/wells/',well_label,'/'));
                recordings = {recordings.Links.Name};
                
                fileObjects(length(recordings),1) = mxw.dataFunctions.fileInterface_2();
                
                for recording = 1:length(recordings)
                    rec_label = recordings{recording};
                    fileObjects(recording) = mxw.dataFunctions.fileInterface_2(char(this.fileNameList(1)),well_label,rec_label);
                end
                
                this.nRecordings = length(recordings);
                
            else
                error('Unrecognized data version.');
            end
            
            this.fileObj = fileObjects;
        end
        
        function compileRawMaps(this)
            
            rawMaps(this.nRecordings).map = [];
            rawMaps(this.nRecordings).spikes = [];
            
            for iRec = 1:this.nRecordings
                rawMaps(iRec).map = this.fileObj(iRec).map;
                rawMaps(iRec).spikes = this.fileObj(iRec).spikes;
            end
            
            this.rawMap = rawMaps;
        end
        
        function computeMap(this)
            
            mapSize = 26400; %%Add this as a parameter in the future
            
            procMaps.xpos = NaN(mapSize, 1);
            procMaps.ypos = NaN(mapSize, 1);
            procMaps.electrode = NaN(mapSize, 1);
            procMaps.recordingIndex = cell(mapSize, 1);
            
            for iRec = 1:this.nRecordings
                for i = 1:length(this.rawMap(iRec).map.channel)
                    electrode = this.rawMap(iRec).map.electrode(i);
                    electrodeIndex = electrode+1;
                    procMaps.electrode(electrodeIndex) = electrode;
                    procMaps.xpos(electrodeIndex) = this.rawMap(iRec).map.x(i);
                    procMaps.ypos(electrodeIndex) = this.rawMap(iRec).map.y(i);
                    procMaps.recordingIndex{electrodeIndex} = [procMaps.recordingIndex{electrodeIndex}, iRec];
                end
            end
            
            procMaps.nonRoutedElec = find(isnan(procMaps.electrode))-1;
            this.processedMap = procMaps;
        end
        
        function compileSpikes(this)
            
            mapSize = 26400; %%Add this as a parameter in the future
            
            spikes.frameno = cell(mapSize, 1);
            spikes.amplitude = cell(mapSize, 1);
            allSpikes(this.nRecordings) = spikes;
            
            for iRec = 1:this.nRecordings
                spikes.frameno = cell(mapSize, 1);
                spikes.amplitude = cell(mapSize, 1);
                
                for i = 1:length(this.rawMap(iRec).map.channel)
                    electrode = this.rawMap(iRec).map.electrode(i);
                    electrodeIndex = electrode+1;
                    channel = this.rawMap(iRec).map.channel(i);
                    idx = find(this.rawMap(iRec).spikes.channel == channel);
                    
                    if ~isempty(idx)
                        spikes.frameno{electrodeIndex} = this.rawMap(iRec).spikes.frameno(idx)';
                        spikes.amplitude{electrodeIndex} = this.rawMap(iRec).spikes.amplitude(idx)';
                    end
                end
                
                allSpikes(iRec) = spikes;
            end
            
            this.extractedSpikes = allSpikes;
        end
        
        function cleanNonRoutedElec(this)
            
            deleteIndex = this.processedMap.nonRoutedElec+1;
            this.processedMap.xpos(deleteIndex) = [];
            this.processedMap.ypos(deleteIndex) = [];
            this.processedMap.recordingIndex(deleteIndex) = [];
            this.processedMap.electrode(deleteIndex) = [];
            
            for iRec = 1:this.nRecordings
                this.extractedSpikes(iRec).frameno(deleteIndex) = [];
                this.extractedSpikes(iRec).amplitude(deleteIndex) = [];
            end
        end
        
        function [recordingsArray] = recordingsArrays(this, electrodes)
            
            index = mod(find(this.processedMap.electrode == electrodes), length(this.processedMap.electrode));
            index(index == 0) = length(this.processedMap.electrode);
            recordingsArray = this.processedMap.recordingIndex(index)';
        end
    end
    
    methods (Access = public)
        
        function removeSpikes(this, indToRemove)
            
            % remove specific spikes from the data
            
            for iRec = 1:this.nRecordings
                
                spikes = this.fileObj(iRec).spikes;
                
                spikes.amplitude(indToRemove) = [];
                spikes.channel(indToRemove) = [];
                spikes.frameno(indToRemove) = [];
                
                this.fileObj(iRec).spikes = spikes;
                %                 this.fileObj(iRec).dataLenSamples = this.fileObj(iRec).dataLenSamples - sampls;
                
            end
            
            this.compileRawMaps();
            this.computeMap();
            this.compileSpikes();
            this.cleanNonRoutedElec();
            
        end
        
        
        function removeSpikesFirstSamples(this, sampls)
            
            % remove all the spikes from the first [samples] specified
            
            for iRec = 1:this.nRecordings
                
                spikes = this.fileObj(iRec).spikes;
                
                ts=spikes.frameno-spikes.frameno(1);
                
                indToRemove = find(ts<sampls);
                
                spikes.amplitude(indToRemove) = [];
                spikes.channel(indToRemove) = [];
                spikes.frameno(indToRemove) = [];
                
                this.fileObj(iRec).spikes = spikes;
                this.fileObj(iRec).dataLenSamples = this.fileObj(iRec).dataLenSamples - sampls;
                
            end
            
            this.compileRawMaps();
            this.computeMap();
            this.compileSpikes();
            this.cleanNonRoutedElec();
            
        end
        
        function removeSpikeArtifacts(this, amp_thr, pre_post_smpls)
            
            % remove all the spikes from the first [samples] specified
            
            for iRec = 1:this.nRecordings
                
                % step 1: load the spikes
                spikes = this.fileObj(iRec).spikes;
                
                % step 2: identify spikes above threshold
                indBadSpikes = find(abs(spikes.amplitude)>amp_thr);
                
                while ~isempty(indBadSpikes)
                    
                    ts=spikes.frameno-spikes.frameno(1);
                    ts_tmp = ts(indBadSpikes(1));
                    
                    indToRemove = find(ts>(ts_tmp-pre_post_smpls) & ts<(ts_tmp+pre_post_smpls));
                    
                    spikes.amplitude(indToRemove) = [];
                    spikes.channel(indToRemove) = [];
                    spikes.frameno(indToRemove) = [];
                    
                    
                    indBadSpikes = find(abs(spikes.amplitude)>amp_thr);
                end
                
                
                
                this.fileObj(iRec).spikes = spikes;
                
            end
            
            this.compileRawMaps();
            this.computeMap();
            this.compileSpikes();
            this.cleanNonRoutedElec();
            
        end
        
        function removeElectrolysisArtifacts(this, nr_of_spikes_thr,pre_post_smpls)
            
            % remove all spikes for a certain framenumber, if there are
            % more than nr_of_spikes_thr spikes during this frame
            
            for iRec = 1:this.nRecordings
                
                % step 1: load the spikes
                spikes = this.fileObj(iRec).spikes;
                
                % step 2: find the unique framenumbers and how many spikes they contain
                frameno = spikes.frameno;
                unique_frames = unique(frameno);
                frame_cnt = groupsummary(frameno,frameno,@numel);
                
                % step 3: find the frame numbers that appear more than nr_of_spikes_thr times
                badFrames = unique_frames(find(frame_cnt>nr_of_spikes_thr));
                
                if ~isempty(badFrames)
                    
                    % step 4: add a window of size pre_post_smpls before
                    % and after each bad frame, which will also be removed
                    framesToRemove = [];
                    
                    for i = 1:length(badFrames)
                        tmp_frame = badFrames(i);
                        framesToRemove = [framesToRemove,tmp_frame-pre_post_smpls:tmp_frame+pre_post_smpls];
                    end
     
                    framesToRemove = unique(framesToRemove);
                    
                    % step 5: find the indices of our bad frames inside the
                    % frameno array and remove these 
                    indBadFrames = ismember(frameno,framesToRemove);
                    
                    spikes.amplitude(indBadFrames) = [];
                    spikes.channel(indBadFrames) = [];
                    spikes.frameno(indBadFrames) = [];
  
                end
                
                
                this.fileObj(iRec).spikes = spikes;
                
            end
            
            this.compileRawMaps();
            this.computeMap();
            this.compileSpikes();
            this.cleanNonRoutedElec();
        
        end
        
        function [preRed,postRed] = removeRedundantSpikes(this,deltaSamples,deltaDist)
            preRed = [];
            postRed = [];
            % remove all the spikes from the first [samples] specified
            for iFile = 1:this.nFiles
                
                preRed = [preRed, length(this.fileObj(iFile).spikes.frameno)];
                % call here:
                indToRemove = mxw.spikeProcessing.detectRedundantSpikes(this, iFile, deltaSamples,deltaDist); % this
                
                spikes = this.fileObj(iFile).spikes;
                spikes.amplitude(indToRemove) = [];
                spikes.channel(indToRemove) = [];
                spikes.frameno(indToRemove) = [];
                
                this.fileObj(iFile).spikes = spikes;
                this.fileObj(iFile).dataLenSamples = this.fileObj(iFile).dataLenSamples; %define sampls this.fileObj(iFile).dataLenSamples ? sampls
                postRed = [postRed, length(this.fileObj(iFile).spikes.frameno)];
                
            end
            
            %disp(sprint(?Percentage of spikes after reduction: %s?, int2str(postRed*100/preRed)))
            
            this.compileRawMaps();
            this.computeMap();
            this.compileSpikes();
            this.cleanNonRoutedElec();
            
        end
        
        
        function [data, recordingsArray, electrodesArray] = extractRawData(this, start, len, varargin)
            
            p = inputParser;
            
            p.addParameter('electrodes', []);
            p.addParameter('files', 1:this.nRecordings);
            p.parse(varargin{:});
            args = p.Results;
            
            if isempty(args.electrodes)
                if this.nRecordings == 1
                    tempFilesArray = this.recordingsArrays(this.fileObj.map.electrode');
                    [electrodesArray, recordingsArray] = mxw.util.electrodes2Files(this.fileObj.map.electrode', tempFilesArray);
                    data = this.fileObj.extractFullRawData(start, len);
                    
                elseif (length(args.files) == 1)
                    tempFilesArray = this.recordingsArrays(this.fileObj(args.files).map.electrode');
                    [electrodesArray, recordingsArray] = mxw.util.electrodes2Files(this.fileObj(args.files).map.electrode', tempFilesArray);
                    index = mod(find(recordingsArray' == args.files), length(recordingsArray));
                    index(index == 0) = length(recordingsArray);
                    recordingsArray = recordingsArray(index');
                    electrodesArray = electrodesArray(index');
                    data = this.fileObj(recordingsArray).extractFullRawData(start, len);
                    
                else
                    error('only one complete file can be extracted at once');
                end
                
            else
                nonRouted = any(this.processedMap.nonRoutedElec == args.electrodes);
                
                if any(nonRouted)
                    error('electrode(s) %s is(are) not routed', num2str(args.electrodes(nonRouted)));
                end
                
                tempFilesArray = this.recordingsArrays(args.electrodes);
                [electrodesArray, recordingsArray] = mxw.util.electrodes2Files(args.electrodes, tempFilesArray);
                index = mod(find(recordingsArray' == args.files), length(recordingsArray));
                index(index == 0) = length(recordingsArray);
                recordingsArray = recordingsArray(index');
                electrodesArray = electrodesArray(index');
                data = zeros(len, sum(cellfun('length', electrodesArray)));
                dataIndexStart = 1;
                
                for iRec = 1:length(recordingsArray)
                    dataIndexEnd = length(electrodesArray{iRec});
                    data(:, dataIndexStart:dataIndexStart + dataIndexEnd - 1) = ...
                        this.fileObj(recordingsArray(iRec)).extractRawData(start, len, electrodesArray{iRec});
                    dataIndexStart = dataIndexStart + dataIndexEnd;
                end
            end
        end
        
        function [waveformCutOuts, electrodesArray] = extractCutOuts(this, spikesTimePoints, prePointsSpike, postPointsSpike, varargin)
            
            p = inputParser;
            
            p.addParameter('files', 1:this.nRecordings);
            p.parse(varargin{:});
            args = p.Results;
            
            if this.nRecordings == 1
                nChannels = length(this.fileObj.map.channel);
                
            elseif (length(args.files) == 1)
                nChannels = length(this.fileObj(args.files).map.channel);
                
            else
                error('choose one file to extract the cutouts');
            end
            dim1 = (prePointsSpike + postPointsSpike) * nChannels;
            waveformCutOuts = zeros(dim1, length(spikesTimePoints));
            
            for i = 1:length(spikesTimePoints)
                [waveforms, ~, electrodesArray] = this.extractBPFData(spikesTimePoints(i) - prePointsSpike, prePointsSpike + postPointsSpike, varargin{:});
                waveformCutOuts(:, i) = reshape(waveforms, dim1, 1);
            end
        end
        
        function [waveformCutOuts, electrodesArray] = extractRawCutOuts(this, spikesTimePoints, prePointsSpike, postPointsSpike, varargin)
            
            p = inputParser;
            
            p.addParameter('files', 1:this.nRecordings);
            p.parse(varargin{:});
            args = p.Results;
            
            if this.nRecordings == 1
                nChannels = length(this.fileObj.map.channel);
                
            elseif (length(args.recordings) == 1)
                nChannels = length(this.fileObj(args.recordings).map.channel);
                
            else
                error('choose one file to extract the cutouts');
            end
            dim1 = (prePointsSpike + postPointsSpike) * nChannels;
            waveformCutOuts = zeros(dim1, length(spikesTimePoints));
            
            for i = 1:length(spikesTimePoints)
                [waveforms, ~, electrodesArray] = this.extractRawData(spikesTimePoints(i) - prePointsSpike, prePointsSpike + postPointsSpike, varargin{:});
                
                m=mean(waveforms(1:(prePointsSpike*0.1),:));
                waveforms=waveforms-m;
                
                waveformCutOuts(:, i) = reshape(waveforms, dim1, 1);
            end
        end
        
        function [data, recordingsArray, electrodesArray] = extractBPFData(this, start, len, varargin)
            
            [data, recordingsArray, electrodesArray] = this.extractRawData(start, len, varargin{:});
            data = this.bandPassFilter.filter(data);
        end
        
        function [data] = extractDAC(this, varargin)
            if isempty(varargin)
                
                data = this.fileObj.extractDAC(1,this.fileObj.dataLenSamples);
            else
                if length(varargin) == 2
                    data = this.fileObj.extractDAC(varargin{1},varargin{2});
                else
                    error('ExtractDAC requires zero or two input arguments')
                end
            end
            
        end
        
        function modifyBPFilter(this, lowCut, highCut, order)
            
            this.bandPassFilter = mxw.util.bandpass(lowCut, highCut, order);
        end
        
    end
    
    methods (Access = private, Static)
        
        function [lowCut, highCut, order] = defaultBPF()
            
            lowCut = 300;
            highCut = 7000;
            order = 4;
        end
        
    end
end

