classdef bandpass
    % BANDPASS is the class that defines the bandpass filter used when 
    % extracting data through the 'mxw.fileManager' methods 
    % 'extractBPFData' and 'extractCutOuts'. This class is instantiated 
    % automatically when creating the 'mxw.fileManager' object, so it does 
    % not have to be instantiated by the user. In order to change the
    % bandpass filter, use the method 'modifyBPFilter' on the 
    % 'mxw.fileManager' object.
    %
    %

    properties
        fs;
        lowcut;
        highcut;
        order;
        nyq;
        low;
        high;
        a;
        b;
    end
    
    methods
        
        function f = bandpass(lowcut, highcut, order)
            f.fs = 20000;
            f.lowcut = lowcut;
            f.highcut = highcut;
            f.order = order;
            f.nyq = 0.5 * f.fs;
            f.low = lowcut   / f.nyq;
            f.high = highcut / f.nyq;
            
%             result = license('test','Signal_Toolbox');
%             if exist('butter')
%             if result
%                 [f.b, f.a] = butter(order, [f.low, f.high], 'bandpass');
%             else
                [f.b, f.a] = mxw.util.get_filter_coeffs(order, f.lowcut, f.highcut, f.fs);
%             end
            
        end
        
        % Apply causal filter
        function Y = filterCausal(obj, X)
            
            Y = filter(obj.b, obj.a, X);
        end
        
        % Apply filter twice to remove phase-shift
        function Y = filter(obj, X)
%             result = license('test','Signal Processing Toolbox');
%             if result
%             if exist('filtfilt')
%                 Y = filtfilt(obj.b, obj.a, X);
%             else
                Y = FiltFiltM(obj.b, obj.a, X);
%             end
        end
    end
end

