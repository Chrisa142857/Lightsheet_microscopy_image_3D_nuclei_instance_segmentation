% Compare volumes
%fname1 =  '/media/SteinLab2/WT7R/output/resampled/transformed/result.tiff';
%fname2 = '/ssd2/userdata/ok37/TOP16R/output/resampled/mask_final/mask_final.tiff';
%fname3 = '/home/ok37/repos/MATLAB/tc_pipeline/supplementary_files/annotation_25-left.tif';

%WT = loadtiff(fname1);
%KO = loadtiff(fname2);
%REF = loadtiff(fname3);

load('/ssd2/userdata/ok37/Top110R/output/variables/I_mask.mat')
df_table = readtable('structure_volumes.csv');
df_table1 = readtable('layer_volumes.csv');

annot_dir = '/home/ok37/repos/MATLAB/tc_pipeline/annotations/cortical_substructures';
annot_dir1 = '/home/ok37/repos/MATLAB/tc_pipeline/annotations/cortical_layers';

annot_dir = dir(annot_dir);
annot_dir = annot_dir(arrayfun(@(x) contains(x.name,'.csv'),annot_dir));

annot_dir1 = dir(annot_dir1);
annot_dir1 = annot_dir1(arrayfun(@(x) contains(x.name,'.csv'),annot_dir1));


mm3 = 0.025*0.025*0.025;

structure_names = ["Agranular Insular","Anterior Cingulate","Auditory","Ectorhinal","Frontal Pole",...
    "Gustatory", "Infralimbic", "Motor","Orbital","Perirhinal","Posterior Parietal","Prelimbic","Retrospinal",...
    "Somatosensory","Temporal","Visual"];

layer_names = ["Layer 1","Layer2/3","Layer 4","Layer 5","Layer 6a","Layer 6b"];


df_out = zeros(1,length(annot_dir));

for i = 1:length(annot_dir)
    df = readtable(annot_dir(i).name);
    disp(annot_dir(i).name)
    ids = df.index;
    
    s = 0;
    for j = 1:length(ids)
        idx = find(I_mask == ids(j));
        s = s + length(idx);
    end
        
    s = s*mm3;
    df_out(i) = s;
end

df_out1 = zeros(1,length(annot_dir1));

for i = 1:length(annot_dir1)
    df1 = readtable(annot_dir1(i).name);
    disp(annot_dir1(i).name)
    ids = df1.index;
    
    s1 = 0;
    for j = 1:length(ids)
        idx = find(I_mask == ids(j));
        s1 = s1 + length(idx);
    end
        
    s1 = s1*mm3;
    df1_out(i) = s1;
end



%df_table = table(structure_names',df_out');
%df_table1 = table(layer_names',df1_out');


df_table(:,end+1) = table(df_out');
df_table1(:,end+1) = table(df1_out');

writetable(df_table,'structure_volumes.csv')
writetable(df_table1,'layer_volumes.csv')



%%



a = dir(pwd);

%Check .tif in current folder
a = a(arrayfun(@(x) contains(x.name,'.csv'),a));
pixel_num = zeros(3,length(a));

img_index = 159;

sample_img = ones([size(REF(:,:,img_index)),3]);
   
structure_table = readtable(a(end).name,'Delimiter',',');
idx = table2array(structure_table(:,1));
[x, y] = find(ismember(REF(:,:,img_index),idx));

for j = 1:length(x)
    sample_img(x(j),y(j),:) = 0.4;
end


for i = 2:length(a)-1   
   structure_table = readtable(a(i).name,'Delimiter',',');
   
   
   %idx = table2cell(structure_table(:,1));
   %idx = cellfun(@(x) str2double(x),idx,'UniformOutput',false);
   idx = table2array(structure_table(:,1));
   
   pixel_num(1,i) = sum(ismember(WT(:),idx))*mm3;
   pixel_num(2,i) = sum(ismember(KO(:),idx))*mm3;
   pixel_num(3,i) = pixel_num(2,i)/pixel_num(1,i);
   
   R = table2array(structure_table(1,2));
   G = table2array(structure_table(1,3));
   B = table2array(structure_table(1,4));
   
    [x, y] = find(ismember(REF(:,:,img_index),idx));
    for j = 1:length(x)
        sample_img(x(j),y(j),1) = R/255;
        sample_img(x(j),y(j),2) = G/255;
        sample_img(x(j), y(j),3) = B/255;
        
    end
       
       
       
       
end


    imshow(sample_img)




