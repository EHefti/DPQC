function lineNumber = findStringInFile(fname,stringToFind)

%FINDSTRINGINFILE Searches for string in a File
%   lineNumber = findStringInFile(fname,stringToFind) 


c=textread(fname,'%s','delimiter','\n');
lineNumber = find(~cellfun(@isempty,strfind(c,stringToFind)));

if isempty(lineNumber)
    error('string not found');
end

if length(lineNumber)>1
    
    warning(['More than one occurences of ' stringToFind ' found. Only the first is returned.']);
    lineNumber = lineNumber(1);
    
end

end