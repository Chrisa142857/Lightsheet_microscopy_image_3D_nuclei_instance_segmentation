function combine_densities(config,count_results,vol_results,dense_results)
%--------------------------------------------------------------------------
% Calculate cell densities for annotated structures based on cell count and
% structure volume results. 
%--------------------------------------------------------------------------

% Load counts and volumes
if isfile(count_results)
    counts = readtable(count_results,'PreserveVariableNames',true);
else
    return
end
if isfile(vol_results)
    volumes = readtable(vol_results,'PreserveVariableNames',true);
else
    return
end
count_names = string(counts.Properties.VariableNames);

% Calculate densities
densities = counts;
for i = 1:length(config.samples)
    vals = volumes.(strcat(config.samples(i),"_Volume"));
    c_idx = contains(count_names,config.samples(i));
    densities(:,c_idx).Variables = densities(:,c_idx).Variables./vals;
end

writetable(densities,dense_results)

end