function [img_directory, output_directory, group] = NM_samples(sample, save_flag)
%--------------------------------------------------------------------------
% Record sample specific information here. Additionally, any default
% processing and/or analysis parameters can be overwritten as this function
% is called after defaults are loaded.
%--------------------------------------------------------------------------

switch sample
    case 'SAMPLE1'
        %img_directory =
        %output_directory =
        %group =
        %channel_num =
        %markers =
        %position_exp = ["[\d*", "\d*]","Z\d*"];        
        %resolution = 
        %ls_width =
        %overlap =
        %orientation =
        %hemisphere =
            
    case 'TEST1'
        % Image directory can be specified as single location if able to
        % dismabiguate channels or as multiple locations pointing to each
        % individual channel
        img_directory = "/test_images";                 % Input image directory        
        output_directory = "/test_images/output";       % Directory to save results
        use_processed_images = false;
        group = ["TEST","WT","R1"];                     % Group name/id
        
        % Specify markers or channel_num or both based on which one is specified in the filename
        channel_num = ["C01","C00"];                         % Channel id
        markers = ["topro","ctip2"];                         % Name of markers present
        
        % If one tile, row/slice positions will be ignored if expression is not located in the filename
        position_exp = ["[\d*", "\d*]","Z\d*"];              % 1x3 string of regular expression specifying image row(y), column(x), slice(z)
        
        % If different resolutions, specify as cell array for each marker
        % (i.e. {[1.21, 1.21, 4], [3.86, 3.86, 4]} 
        resolution = [1.21, 1.21, 4];                        % Image reolution in um/voxel
        ls_width = 50;                                       % Light sheet width as percentage
        overlap = 0.15;                                      % Overlap between tiles as fraction
        
        % Orientation key: anterior(a)/posterior(p), superior(s)/inferior(i), left(l)/right(r)
        % Specified as location at row(y)=0, column(x)=0, slice(z)=0
        orientation = "ail";                                 % 1x3 string specifying sample orientation
        hemisphere = "left";                                 % "left","right","both","none"
        
    otherwise
        error("Sample %s does not exist in NM_samples.",sample)
end

%--------------------------------------------------------------------------
% Do not edit
% Append sample info to variable structure
sample_id = sample;
if save_flag
    save(fullfile('data','tmp','NM_variables.mat'),'-mat','-append')
end
end

