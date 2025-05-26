function resultMatrix = createPathMatrix(baseRoot, patternParts)
%
%   Inputs:
%       baseRoot    - A string specifying the root path where the search begins.
%       patternParts - A cell array of strings defining the folder structure
%                      pattern, including wildcards (*). This pattern should
%                      be designed such that the 'Week' and 'Plate'
%                      information can be extracted from the matching paths.
%
%   Output:
%       resultMatrix - A cell array matrix.
%                      Column 1: Full path (string)
%                      Column 2: Week (date string, extracted from path)
%                      Column 3: Plate (string, extracted from path)

    currentMatchingPaths = {baseRoot}; % Start with the root
    
    % This will hold all the final paths to process for output
    finalPathsOnly = {}; 

    for i = 1:length(patternParts)
        currentPatternPart = patternParts{i};
        %fprintf('%s \n', currentPatternPart);
        nextLevelPaths = {}; % To store paths for the next iteration
        
        for j = 1:length(currentMatchingPaths)
            currentPath = currentMatchingPaths{j};
                        
            d = dir(currentPath);
            %fprintf('Found %d directories \n', size(d));
            
            % Filter for actual directories (not files or '.'/'..')
            subDirs = d([d.isdir] & ~strcmp({d.name}, '.') & ~strcmp({d.name}, '..'));
            
            for k = 1:length(subDirs)
                folderName = subDirs(k).name;
                parentPath = subDirs(k).folder;
                if regexp(folderName, currentPatternPart)
                    %fprintf('%s \n', subDirs(k).name);
                    fullPathForNextLevel = fullfile(parentPath, folderName);
                    nextLevelPaths{end+1} = fullPathForNextLevel; %#ok<AGROW>
                end
            end
        end
        currentMatchingPaths = nextLevelPaths; % Update for the next level of pattern
        %fprintf('We have %d current Matching Paths. \n', length(currentMatchingPaths));

        % If this is the last pattern part, these are our final full paths
        if i == length(patternParts)
            finalPathsOnly = currentMatchingPaths;
        end
    end
    %fprintf('Found %d final paths \n', length(finalPathsOnly));
    
    % Initialize the result matrix
    resultMatrix = cell(length(finalPathsOnly), 3); % Pre-allocate space

    % Extract information for each final path
    for pIdx = 1:length(finalPathsOnly)
        currentFullPath = finalPathsOnly{pIdx};
        
        % Split the path into parts
        pathParts = strsplit(currentFullPath, filesep); % filesep handles '/' or '\'

        % Find the index of last element of the baseRoot in the pathParts to get relative indices
        baseRootParts = strsplit(baseRoot, filesep);
        baseRootParts = baseRootParts(~cellfun('isempty', baseRootParts));
        matchIdx = find(strcmp(pathParts, baseRootParts{end}), 1, 'last');
        
        % Using the direct index based on the example structure:
        weekFolder = pathParts{matchIdx + 1}; % This is the first element in the pattern
        plateFolder = pathParts{matchIdx + 2}; % This is the second element in the pattern

        resultMatrix{pIdx, 1} = currentFullPath;
        resultMatrix{pIdx, 2} = weekFolder;
        resultMatrix{pIdx, 3} = plateFolder;
    end
    
    % Display a summary
    fprintf('Found %d paths matching the pattern.\n', size(resultMatrix, 1));
    for pathIdx = 1:size(resultMatrix, 1)
        fprintf('  Path %d: %s \n', pathIdx, resultMatrix{pathIdx, 1});
    end
end
