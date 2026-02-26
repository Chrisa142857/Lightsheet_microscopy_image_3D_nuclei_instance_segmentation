
%% Evaluate 3DUnet performace
% Evaluation options
evaluation_method = '3dunet';
resolution = '121';
true_directory = 'true_filled';
trim_edges = 'true';
threshold = 0.5;
edge_threshold = 0.25;
create_performance_img = false;
padding = [3,3,3];
use_watershed = true;
use_imopen = false;
conn_comp = 26;
int_thresh = 1100;

%%%%%%
addpath(genpath('functions'))
files = dir(fullfile('data',true_directory,resolution));
f_files = files(arrayfun(@(s) s.name(1) == 'f',files));
l_files = files(arrayfun(@(s) s.name(1) == 'l',files));

precision = zeros(1,length(l_files));
recall = zeros(1,length(l_files));
uncertainty = zeros(1,length(l_files));

% Nearest Centroid Distance
for i = 1:length(l_files)
    % Read raw label and data files
    data = niftiread(fullfile(files(i).folder,f_files(i).name));
    label = niftiread(fullfile(files(i).folder,l_files(i).name));
    
    % Choose evaluation method
    switch evaluation_method
        case '3dunet'
            % Calculate 3DUnet Predicted Centroids
            % 3DUnet assumed to given as probability image that needs to be
            % binarized and centroids calculated
            prediction_location = fullfile(pwd,'results','validation_results',...
                sprintf('f%d',i),'prediction.nii');
            
            % Read and trim prediction
            prediction = niftiread(prediction_location);

            % If 4 dimensions, subtract edge binary from fill binary
            if ndims(prediction) == 4
                fill_pred = squeeze(prediction(1,:,:,:));
                edge_pred = squeeze(prediction(2,:,:,:));
                
                edge_pred = imbinarize(edge_pred,edge_threshold);
                prediction = fill_pred - (edge_pred*0.75);
            end
            
            % Uncertainty given as pixels where probability is between 5%
            % and 95%
            uncertainty(i) = sum(prediction(:)>0.05 & prediction(:)<0.95)/numel(prediction)*100;
            
            % Clean using morphological opening
            if use_imopen
                se2 = strel([1 1;1 1]);
                for z = 1:size(prediction,3)
                    prediction(:,:,z) = imopen(prediction(:,:,z),se2);
                end
            end
            
            % Binarize prediction
            prediction = imbinarize(prediction,threshold);
            
            % Apply watershed if necessary
            if use_watershed
                prediction = use_2d_watershed(prediction);
            end

            % Connect components
            cc_predict = bwconncomp(prediction,conn_comp);

            % Calculate centroids
            [x,y,z] = cellfun(@(s) ind2sub(cc_predict.ImageSize,s),...
                cc_predict.PixelIdxList,'UniformOutput',false);
            unet_centroids = cellfun(@(s,t,u) [mean(s) mean(t) mean(u)],...
                x,y,z,'UniformOutput',false);
            centroids = round(cell2mat(unet_centroids'));

        case 'hessian'
            % Calculate Hessian Predicted Centroids
            centroids = hessian_centroid(data);
        case 'watershed'
            % Calculate Watershed Predicted Centroids
            centroids = watershed_centroid(data);
        case 'cubic'
            % Cubic centroids are predicted seperately using provided
            % python code. Only centroids are loaded
            % Rotate and flip images to match orientation
            if i == 1
                results_path = fullfile(pwd,'Updated Training Samples', ...
                    'CUBIC','results');
                addpath(genpath(results_path))
            end
            label = imrotate(label,90);
            data = imrotate(data,90);
            data = flip(data,1);
            label = flip(label,1); 
    
            centroids = cubic_centroid(i,resolution);
            centroids = round(centroids);
            
            % Create arbitrary prediction image. Doesn't affect total error
            % rate
            prediction = zeros(size(label));
            for j = 1:size(centroids,1)
               pos = centroids(j,:);
               prediction(pos(1),pos(2),pos(3)) = 1;
            end
            se = strel([1, 1, 1;1,1,1;1,1,1]);
            prediction = imdilate(prediction,se);
            
        case 'clearmap'
            % Clearmap centroids are predicted seperately using provided
            % clearmap code. Only centroids are loaded
            % Rotate and flip images to match orientation
            if i == 1
                results_path = fullfile(pwd,'Updated Training Samples', ...
                    'ClearMap','results');
                addpath(genpath(results_path))
            end
            label = imrotate(label,90);
            data = imrotate(data,90);
            data = flip(data,1);
            label = flip(label,1); 
            
            centroids = clearmap_centroid(i,resolution);
            centroids = round(centroids);
                        % Create arbitrary prediction image. Doesn't affect total error
            % rate
            prediction = zeros(size(label));
            for j = 1:size(centroids,1)
               pos = centroids(j,:);
               prediction(pos(1),pos(2),pos(3)) = 1;
            end
            se = strel([1, 1, 1;1,1,1;1,1,1]);
            prediction = imdilate(prediction,se);
            
        case 'filled1'
            % Old test with 3D filled objects
            % Needs updating
            % Calculate 3DUnet Predicted Centroids
            prediction_location = fullfile(pwd,'prediction',...
                sprintf('f%d',i),'prediction.nii.gz');
            
            prediction = niftiread(prediction_location);
            prediction_raw = prediction;
            %prediction = uint8(prediction == 1);
            prediction = imbinarize(prediction,threshold);
            
            if isequal(use_watershed,'true')
                prediction = use_2d_watershed(prediction);
            end
            
            cc_predict = bwconncomp(prediction,26);

            [x,y,z] = cellfun(@(s) ind2sub(cc_predict.ImageSize,s),cc_predict.PixelIdxList,'UniformOutput',false);
            unet_centroids = cellfun(@(s,t,u) [mean(s) mean(t) mean(u)],x,y,z,'UniformOutput',false);
            centroids = round(cell2mat(unet_centroids'));
            
            label_path = fullfile(pwd,'Labels','l5_OKu-Training-Side60-[01x01]-OKu-TL-SB_7_4_lbl');
            label = niftiread(label_path);
            label = imresize3(label, size(prediction),'Method','nearest');
            
            label = flip(label,1);
            
            if isequal(trim_edges,'true')
                label = trim_nuclei_around_edges(label, padding);
            end
    end
    
    % Trim prediction image
    % Trim label images to area with labels
    [label, edge] = trim_to_labels(double(label));
    data = data(edge(1):edge(2),edge(3):edge(4),edge(5):edge(6));
    prediction = prediction(edge(1):edge(2),edge(3):edge(4),edge(5):edge(6));
    if isequal(trim_edges,'true')
        label = trim_nuclei_around_edges(label, padding);
    end
    
    % Trim and readjust centroids
    centroids = centroids(centroids(:,1)>edge(1)-1 & centroids(:,1)<edge(2)+1,:);
    centroids = centroids(centroids(:,2)>edge(3)-1 & centroids(:,2)<edge(4)+1,:);
    centroids = centroids(centroids(:,3)>edge(5)-1 & centroids(:,3)<edge(6)+1,:);
    centroids(:,1) = centroids(:,1) - edge(1)+1;
    centroids(:,2) = centroids(:,2) - edge(3)+1;
    centroids(:,3) = centroids(:,3) - edge(5)+1;
    cen_total = size(centroids,1);
    
    % Remove centroids with low intensity
    for k = 1:size(data,3)
	data(:,:,k) = imgaussfilt(data(:,:,k),1);
    end
    if int_thresh > 0
        for j = 1:size(centroids,1)
           pos = centroids(j,:);
           int = data(pos(1),pos(2),pos(3));
            if mean(int) < int_thresh
            centroids(j,:) = NaN;
            end
        end
       centroids = centroids(~isnan(centroids(:,1)),:);
    end   
    
    % Get unqiue labels
    uniq_lbls = unique(label);
    uniq_lbls = length(uniq_lbls(uniq_lbls > 0));
    
    % Calculate errors
    [mult(i), miss(i), merge(i), empty(i), correct(i), total_error(i), mult_lbls,...
        miss_lbls, merge_lbls] = evaluate_by_fill(centroids,...
        label,prediction);
    
    % Calculate precision and recall
    precision(i) = correct(i)/(correct(i) + mult(i) + empty(i));
    recall(i) = correct(i)/(correct(i) + merge(i) + miss(i));
    
    % Give error rates as percentage
    mult(i) = mult(i)/uniq_lbls;
    miss(i) = miss(i)/uniq_lbls;
    merge(i) = merge(i)/uniq_lbls;
    
    % Create performance image
    if create_performance_img
        save_img = create_performance_image(label,data,centroids, ...
            mult_lbls,miss_lbls, merge_lbls);

        save_name = sprintf('Overlay_%s.tif',f_files(i).name);
        imwrite(squeeze(save_img(:,:,1,:)),save_name)
        for j = 2:size(save_img,3)
            imwrite(squeeze(save_img(:,:,j,:)),save_name,'WriteMode','append'); 
        end
    end
end

%fprintf('\nPrecision: %f\n',mean(precision))
%fprintf('Recall: %f\n',mean(recall))
fprintf('\nMultiple Centroids: %f\n',mean(mult))
fprintf('Missed Centroids: %f\n',mean(miss))
fprintf('Merged Centroids: %f\n',mean(merge))
fprintf('Total Error: %f\n',mean(total_error))
fprintf('Uncertainty: %f\n',mean(uncertainty))
disp(precision)
disp(recall)
