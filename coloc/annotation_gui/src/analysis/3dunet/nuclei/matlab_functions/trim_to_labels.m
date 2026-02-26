function [labels_cropped, edge_pos] = trim_to_labels(labels)
    % Find x,y positions where labels appear
    [x,y] = find(sum(labels,3)>0);
    x_left = min(x);
    x_right = max(x);
    y_left = min(y);
    y_right = max(y);

    % Find z positions where labels appear
    z = find(sum(sum(labels,1),2)>0);
    z_left = min(z);
    z_right = max(z);
    
    % Trim labels
    labels_cropped = labels(y_left:y_right,x_left:x_right,z_left:z_right);
    
    % Store edge positions
    edge_pos = [y_left, y_right, x_left, x_right, z_left, z_right];
end