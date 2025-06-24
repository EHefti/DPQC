function [ os ] = detect()
    % DETECT is a function that determines the operating system in use.
    % After determining the operating system, DETECT sets the operating
    % system functions to: delete a folder, copy a folder, and so on. In
    % order to add new functions, just add them under the operating system
    % cases. 
    % 
    % mxw.util.os.detect();
    %
    %

if ismac
    % Code to run on Mac plaform
    os.delete = 'rm ';
    os.copy = 'cp ';
    
elseif isunix
    % On Linux plaform
    os.delete = 'rm ';
    os.copy = 'cp ';
    
elseif ispc
    % On Windows platform
    os.delete = 'del ';
    os.copy = 'copy ';
    
else
    disp('Platform not supported')
end

end

