%% Specify Directory Information
% These are the key parameters

% Specify which structures to compare
results_directory = [];                                 % Output directory for summary (default: numorph/results)
compare_structures_by = "csv";                          % index, csv, depth; Select which structures for multiple comparisons.
structure_csv = "harris_cortical_groupings_w_iso.csv";  % csv file containing structures to compare if comparing by .csv
structure_depth = [];                                   % minimum structure graph order for comparisons (1-10) if comparing by depth

% For cell counting, specify cell-type class information
use_classes = "true";                   % Use cell-type classes. Otherwise raw centroid counts will be used
keep_classes = 1:3;                     % If using classes, select which ones to evaluate
class_names = [];                       % Assign names to classes
minimum_cell_number = 1;                % Minimum number of cells for each structure to be compared

% Specify groups for 
sub_groups = ["WT","KO"];               % Select groups to compare. Defined by 2nd element in sample 'group' variable
paired = "false";                       % Performed paired t-test. Samples paired by 3rd element in sample 'group' variable
