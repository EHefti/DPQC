function allMatchingPaths = findAllActivityScans(baseRoot, patternParts)
%   Inputs:
%       baseRoot: The path were all folders are in that should be analysed
%       patternParts: The folder structure of the Activity Scans
%
%   Output:
%       allMatchingPaths: A list of all paths that match the pattern

    allMatchingPaths = {baseRoot}; % Start with the root
    
    for i = 1:length(patternParts)
        currentPatternPart = patternParts{i};
        nextLevelPaths = {}; % To store paths for the next iteration
    
        for j = 1:length(allMatchingPaths)
            currentBasePath = allMatchingPaths{j};
            
            searchDirPattern = fullfile(currentBasePath, currentPatternPart);
            
            d = dir(searchDirPattern);
            
            % Filter for actual directories (not files or '.'/'..')
            subDirs = d([d.isdir] & ~strcmp({d.name}, '.') & ~strcmp({d.name}, '..'));
            
            for k = 1:length(subDirs)
                nextLevelPaths{end+1} = fullfile(currentBasePath, subDirs(k).name); %#ok<AGROW>
            end
        end
        allMatchingPaths = nextLevelPaths; % Update for the next level of pattern
        
        if isempty(allMatchingPaths) && i < length(patternParts)
            fprintf('Early exit: No paths found for pattern part "%s"\n', currentPatternPart);
            break; % No matches, no need to continue
        end
    end
    
    disp('Found directories matching the full pattern:');
    if isempty(allMatchingPaths)
        disp('None.');
    else
        disp(allMatchingPaths');
    end
end
