classdef neuronManager
    %NEURONMANAGER is the main class for handling neurons and neuron-related
    % functions
    %   Detailed explanation goes here
    
    properties
        
        % file info
        fname
        Fs
        
        % ts info
        ts
        frame_no
        ts_fidx % if ts from multiple files
        
        % template and waveforms
        template
        traces
        twoDfilteredTemplate
        
        
        % electrode info
        x
        y
        electrode
        channel
        source_el
        neighbours
        
    end
    
    methods
        function obj = neuronManager(n_input)
            %NEURONMANAGER Construct an neuron object
            %   Detailed explanation goes here
            
            if iscell(n_input)
                %                 obj = cell(size(n_input));
                
                for i=1:length(n_input)
                    obj(i) = mxw.neuronManager(n_input{i});
                end
            else
                
                % file info
                obj.fname = n_input.fileObj.filePath;
                obj.Fs = n_input.fileObj.samplingFreq;
                
                % ts info
                obj.ts = double(round(n_input.ts));
                obj.frame_no = n_input.frame_no;
                
                % obj.ts_fidx % if ts from multiple files
                
                % electrode info
                obj.x = n_input.fileObj.map.x;
                obj.y = n_input.fileObj.map.y;
                obj.electrode = n_input.fileObj.map.electrode;
                obj.channel = n_input.fileObj.map.channel;
                obj.source_el = n_input.sortEls(1);
            end
        end
        
        function obj = extractTemplate(obj,varargin)
            
            %extractTemplate extract the template (footprint) of a neuron
            %
            %  extractTemplate computes the STA extracellular footprint
            %
            %  extractTemplate('pre',20,'post',40) defines how many samples
            %  before and after the event should be cut. Default: pre = 30
            %  / post = 50
            %
            
            
            p = inputParser;
            
            p.addParameter('pre', 20);
            p.addParameter('post', 40);
            p.parse(varargin{:});
            args = p.Results;
            
            if length(obj)>1
                for i=1:length(obj)
                    obj(i) = obj(i).extractTemplate(args);
                end
            else
                
                
                datainfo = mxw.fileManager(obj.fname);
                
                [waveformCutOuts, electrodesArray] = datainfo.extractCutOuts(obj.ts, args.pre, args.post);
                
                t=mean(waveformCutOuts,2);
                obj.template=reshape(t,[args.pre+args.post,length(electrodesArray{1})]);
            end
        end
        
        function obj = extractTraces(obj,varargin)
            
            %EXTRACTTRACES Extract the traces waveforms
            %
            %  extractTraces('pre',20,'post',40) defines how many samples
            %  before and after the event should be cut. Default: pre = 30
            %  / post = 50
            %
            %  extractTraces('tracesNumber',100) limit number of extracted
            %  Traces
            %
            %  extractTraces('tsIdx',[1:5 15 22]) only extract traces from
            %  defined spike times
            %
            % See also: extractTemplate
            
            
            
            p = inputParser;
            
            p.addParameter('pre', 20);
            p.addParameter('post', 50);
            p.addParameter('tracesNumber', []);
            p.addParameter('tsIdx', []);
            p.parse(varargin{:});
            args = p.Results;
            
            if length(obj)>1
                h = waitbar(0,'Traces are loaded...');
                for i=1:length(obj)
                    waitbar(i / length(obj))
                    obj(i) = obj(i).extractTraces(args);
                end
                close(h)
            else
                
                
                pre = args.pre;
                post = args.post;
                
                
                datainfo = mxw.fileManager(obj.fname);
                
                [waveformCutOuts, electrodesArray] = datainfo.extractCutOuts(obj.ts, pre, post);
                
                if isempty(obj.template)
                    t=mean(waveformCutOuts,2);
                    obj.template=reshape(t,[args.pre+args.post,length(electrodesArray{1})]);
                end
                maxTs = size(waveformCutOuts,2);
                
                if isempty(args.tracesNumber)
                    ts_ind = 1:maxTs;
                else
                    ts_ind = 1:min(args.tracesNumber,maxTs);
                end
                
                if ~isempty(args.tsIdx)
                    ts_ind = args.tsIdx;
                end
                
                tr=cell(size(electrodesArray{1}));
                for i= 1:length(tr)
                    start = (i-1)*(pre+post)+1;
                    ind = start:(start+pre+post-1);
                    tr{i}=waveformCutOuts(ind,ts_ind);
                end
                
                obj.traces = tr;
                
            end
        end
        
        
        function obj = extractRawTemplate(obj,varargin)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            %             outputArg = obj.Property1 + inputArg;
            
            
            p = inputParser;
            
            p.addParameter('pre', 3000);
            p.addParameter('post', 12000);
            p.parse(varargin{:});
            args = p.Results;
            
            
            if length(obj)>1
                for i=1:length(obj)
                    obj(i) = obj(i).extractRawTemplate(args);
                end
            else
                
                pre = args.pre;
                post = args.post;
                
                
                datainfo = mxw.fileManager(obj.fname);
                
                [waveformCutOuts, electrodesArray] = datainfo.extractRawCutOuts(obj.ts, pre, post);
                
                
                
                t=mean(waveformCutOuts,2);
                obj.template=reshape(t,[pre+post,length(electrodesArray{1})]);
            end
        end
        
        function obj = extractRawTraces(obj,varargin)
            
            %EXTRACTTRACES Extract the traces waveforms
            %
            %  extractTraces('pre',20,'post',40) defines how many samples
            %  before and after the event should be cut. Default: pre = 30
            %  / post = 50
            %
            %  extractTraces('tracesNumber',100) limit number of extracted
            %  Traces
            %
            %  extractTraces('tsIdx',[1:5 15 22]) only extract traces from
            %  defined spike times
            %
            % See also: extractTemplate
            
            
            
            p = inputParser;
            
            p.addParameter('pre', 20);
            p.addParameter('post', 50);
            p.addParameter('tracesNumber', []);
            p.addParameter('tsIdx', []);
            p.parse(varargin{:});
            args = p.Results;
            
            if length(obj)>1
                h = waitbar(0,'Traces are loaded...');
                for i=1:length(obj)
                    waitbar(i / length(obj))
                    obj(i) = obj(i).extractTraces(args);
                end
                close(h)
            else
                
                
                pre = args.pre;
                post = args.post;
                
                
                datainfo = mxw.fileManager(obj.fname);
                
                [waveformCutOuts, electrodesArray] = datainfo.extractRawCutOuts(obj.ts, pre, post);
                
                if isempty(obj.template)
                    t=mean(waveformCutOuts,2);
                    obj.template=reshape(t,[args.pre+args.post,length(electrodesArray{1})]);
                end
                maxTs = size(waveformCutOuts,2);
                
                if isempty(args.tracesNumber)
                    ts_ind = 1:maxTs;
                else
                    ts_ind = 1:min(args.tracesNumber,maxTs);
                end
                
                if ~isempty(args.tsIdx)
                    ts_ind = args.tsIdx;
                end
                
                tr=cell(size(electrodesArray{1}));
                for i= 1:length(tr)
                    start = (i-1)*(pre+post)+1;
                    ind = start:(start+pre+post-1);
                    tr{i}=waveformCutOuts(ind,ts_ind);
                end
                
                obj.traces = tr;
                
            end
        end
        
        function plot(obj,varargin)
            mxw.plot.neuron(obj,varargin)
        end
    end
end


