function I_mask2 = condense_annotations(I_mask)

annot_dir = '/home/ok37/repos/MATLAB/tc_pipeline/annotations/cortical_substructures';

annot_dir = dir(annot_dir);
annot_dir = annot_dir(arrayfun(@(x) contains(x.name,'.csv'),annot_dir));

I_mask2 = zeros(size(I_mask));

for i = 1:length(annot_dir)
    df = readtable(annot_dir(i).name);
    disp(annot_dir(i).name)
    ids = df.index;
    
    for j = 1:length(ids)
        I_mask2(I_mask == ids(j)) = i;
    end
end

%df_out = zeros(1,length(annot_dir));

%for i = 1:length(annot_dir)
%    df = readtable(annot_dir(i).name);
%    disp(annot_dir(i).name)
%    ids = df.index;
    
%    s = 0;
%    for j = 1:length(ids)
%        idx = find(I_mask == ids(j));
%        s = s + length(idx);
%    end
        
%    s = s*mm3;
%    df_out(i) = s;
%end





end