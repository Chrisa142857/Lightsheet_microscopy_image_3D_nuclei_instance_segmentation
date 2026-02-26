```
L35D719P4_patches
Correctly classified 71.13% of cells
tensor([  0, 356,  261,  220,  139])
Correctly classified 68.57% of class-01 cells
Correctly classified 75.86% of class-02 cells
Correctly classified 73.33% of class-03 cells
Correctly classified 66.67% of class-04 cells
```
python coloc_classifier_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet # 

```
L66D764P8_isocortex_patches
tensor([  0, 605, 220,  63,  34])
Correctly classified 86.96% of cells
Correctly classified 93.44% of class-01 cells
Correctly classified 61.90% of class-02 cells
Correctly classified 100.00% of class-03 cells
Correctly classified 100.00% of class-04 cells
```
python coloc_classifier_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet # 

```
L35D719P4_patches + L66D764P8_isocortex_patches
tensor([  0, 356, 261, 220, 139])
tensor([  0, 961, 481, 283, 173])
Correctly classified 76.19% of cells
Correctly classified 93.33% of class-01 cells
Correctly classified 67.35% of class-02 cells
Correctly classified 68.00% of class-03 cells
Correctly classified 40.00% of class-04 cells
```
python coloc_classifier_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet # 



```
L66D764P8_isocortex_patches
tensor([  0, 605, 220,  63,  34])
Correctly classified 84.78% of cells
Correctly classified 86.89% of class-01 cells
Correctly classified 85.71% of class-02 cells
Correctly classified 40.00% of class-03 cells
Correctly classified 100.00% of class-04 cells
```
python coloc_classifierBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet --device cuda:1 # 


```
L35D719P4_patches + L66D764P8_isocortex_patches
Correctly classified 78.31% of cells
Correctly classified 88.89% of class-01 cells
Correctly classified 69.39% of class-02 cells
Correctly classified 76.00% of class-03 cells
Correctly classified 60.00% of class-04 cells
```
python coloc_classifier_multiclass_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass --device cuda:1 --nologging


```
L66D764P8_isocortex_patches
Correctly classified 86.96% of cells
Correctly classified 90.16% of class-01 cells
Correctly classified 76.19% of class-02 cells
Correctly classified 80.00% of class-03 cells
Correctly classified 100.00% of class-04 cells
```
python coloc_classifier_multiclass_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass --device cuda:5 --nologging
python coloc_classifier_multiclass_trainval.py --proj_name coloc-p4-resnet-multiclass --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass_model_weights_2025-07-16-11-16-33.387311_bestACC-86.96.pth


```
L66D764P8_isocortex_patches
Correctly classified 89.13% of cells
Correctly classified 86.89% of class-01 cells
Correctly classified 90.48% of class-02 cells
Correctly classified 100.00% of class-03 cells
Correctly classified 100.00% of class-04 cells
```
python coloc_classifier_multiclassBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox --device cuda:5 --nologging
python coloc_classifier_multiclassBbox_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox_model_weights_2025-07-16-12-12-28.125807_bestACC-89.13.pth




```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7
tensor([  0, 605, 220,  63,  34], [  0, 554, 227,  97,  16], [  0, 438, 178, 109,  23]
Correctly classified 90.62% of cells
Correctly classified 94.48% of class-01 cells
Correctly classified 85.45% of class-02 cells
Correctly classified 78.12% of class-03 cells
Correctly classified 100.00% of class-04 cells
```
python coloc_classifier_multiclassBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-3brain --device cuda:5 --nologging
python coloc_classifier_multiclassBbox_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-3brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-3brain_model_weights_2025-08-04-12-05-28.957193_bestACC-90.62.pth



```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7
tensor([  0, 605, 220,  63,  34], [  0, 554, 227,  97,  16], [  0, 438, 178, 109,  23]
Correctly classified 89.06% of cells
Correctly classified 97.55% of class-01 cells
Correctly classified 67.27% of class-02 cells
Correctly classified 81.25% of class-03 cells
Correctly classified 100.00% of class-04 cells
```
python coloc_classifier_multiclass_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-3brain --device cuda:5 --nologging
python coloc_classifier_multiclass_trainval.py --proj_name coloc-p4-resnet-multiclass-3brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-3brain_model_weights_2025-08-19-15-16-47.041817_bestACC-89.06.pth



```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000
[  0, 605, 220,  63,  34], [  0, 554, 227,  97,  16], [  0, 438, 178, 109,  23], [  0, 473, 137,  84,   9]
Correctly classified 88.65% of cells
Correctly classified 92.46% of class-01 cells
Correctly classified 83.95% of class-02 cells
Correctly classified 82.50% of class-03 cells
Correctly classified 66.67% of class-04 cells
```
python coloc_classifier_multiclassBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-4brain --device cuda:7 --nologging
python coloc_classifier_multiclassBbox_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-4brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-4brain_model_weights_2025-08-28-15-17-04.387584_bestACC-88.65.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 (xydownr=500, zdownr=50)
[  0, 605, 220,  63,  34], [  0, 554, 227,  97,  16], [  0, 438, 178, 109,  23], [  0, 473, 137,  84,   9]
Correctly classified 88.65% of cells
Correctly classified 88.94% of class-01 cells
Correctly classified 95.06% of class-02 cells
Correctly classified 82.50% of class-03 cells
Correctly classified 33.33% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locXYZ-4brain --device cuda:5 --nologging
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locXYZ-4brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locXYZ-4brain_model_weights_2025-08-28-17-23-45.915499_bestACC-88.65.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 (xydownr=200, zdownr=50)
[  0, 605, 220,  63,  34], [  0, 554, 227,  97,  16], [  0, 438, 178, 109,  23], [  0, 473, 137,  84,   9]
Correctly classified 89.57% of cells
Correctly classified 92.46% of class-01 cells
Correctly classified 88.89% of class-02 cells
Correctly classified 80.00% of class-03 cells
Correctly classified 66.67% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr200-4brain --device cuda:4 --nologging
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr200-4brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locXYZdownr200-4brain_model_weights_2025-08-28-17-24-48.427329_bestACC-89.57.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 (xydownr=200, zdownr=50)
[  0, 605, 220,  63,  34], [  0, 554, 227,  97,  16], [  0, 438, 178, 109,  23], [  0, 473, 137,  84,   9]
Correctly classified 88.65% of cells
Correctly classified 91.96% of class-01 cells
Correctly classified 83.95% of class-02 cells
Correctly classified 85.00% of class-03 cells
Correctly classified 66.67% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr200-4brain --device cuda:4 --nologging --lr_scheduler --focalloss
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr200-4brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locXYZdownr200-4brain_model_weights_2025-08-28-17-54-40.640840_bestACC-88.65.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 (xydownr=200, zdownr=20)
[  0, 605, 220,  63,  34], [  0, 554, 227,  97,  16], [  0, 438, 178, 109,  23], [  0, 473, 137,  84,   9]
Correctly classified 88.65% of cells
Correctly classified 90.95% of class-01 cells
Correctly classified 86.42% of class-02 cells
Correctly classified 85.00% of class-03 cells
Correctly classified 66.67% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr20-4brain --device cuda:5 --nologging --lr_scheduler --focalloss
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr200-4brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locXYZdownr20-4brain_model_weights_2025-08-28-17-56-03.507728_bestACC-88.65.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000
[  0, 605, 220,  63,  34], [  0, 554, 227,  97,  16], [  0, 438, 178, 109,  23], [  0, 473, 137,  84,   9]
Correctly classified 88.96% of cells
Correctly classified 91.46% of class-01 cells
Correctly classified 85.19% of class-02 cells
Correctly classified 87.50% of class-03 cells
Correctly classified 66.67% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locLin-4brain --device cuda:6 --nologging
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locLin-4brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locLin-4brain_model_weights_2025-08-28-17-20-42.169507_bestACC-88.96.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000
[  0, 605, 220,  63,  34], [  0, 554, 227,  97,  16], [  0, 438, 178, 109,  23], [  0, 473, 137,  84,   9]
Correctly classified 89.26% of cells
Correctly classified 91.46% of class-01 cells
Correctly classified 86.42% of class-02 cells
Correctly classified 85.00% of class-03 cells
Correctly classified 83.33% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locLin-4brain --device cuda:7 --nologging --lr_scheduler --focalloss
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locLin-4brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locLin-4brain_model_weights_2025-08-28-17-20-42.337729_bestACC-89.26.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000
[605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22]
Correctly classified 86.32% of cells
Correctly classified 93.16% of class-01 cells
Correctly classified 77.78% of class-02 cells
Correctly classified 77.55% of class-03 cells
Correctly classified 63.64% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locLin-5brain --device cuda:7 --nologging --lr_scheduler --focalloss
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locLin-5brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locLin-5brain_model_weights_2025-08-29-18-00-36.707085_bestACC-86.32.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000
[605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22]
Correctly classified 86.57% of cells
Correctly classified 87.61% of class-01 cells
Correctly classified 92.59% of class-02 cells
Correctly classified 69.39% of class-03 cells
Correctly classified 81.82% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locLin-5brain --device cuda:6 --nologging
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locLin-5brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locLin-5brain_model_weights_2025-08-29-18-21-58.821600_bestACC-86.57.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000
[605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22]
Correctly classified 86.82% of cells
Correctly classified 90.60% of class-01 cells
Correctly classified 80.56% of class-02 cells
Correctly classified 83.67% of class-03 cells
Correctly classified 81.82% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr20-5brain --device cuda:7 --nologging --lr_scheduler --focalloss
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr20-5brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locXYZdownr20-5brain_model_weights_2025-08-30-15-53-27.029736_bestACC-86.82.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000
[605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22]  (xydownr=200, zdownr=50)
Correctly classified 88.06% of cells
Correctly classified 91.88% of class-01 cells
Correctly classified 83.33% of class-02 cells
Correctly classified 81.63% of class-03 cells
Correctly classified 81.82% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr50-5brain --device cuda:6 --nologging --lr_scheduler --focalloss
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr50-5brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locXYZdownr50-5brain_model_weights_2025-08-30-15-54-37.366983_bestACC-88.06.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000
[605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22]  (xydownr=200, zdownr=50)
Correctly classified 87.31% of cells
Correctly classified 91.88% of class-01 cells
Correctly classified 84.26% of class-02 cells
Correctly classified 73.47% of class-03 cells
Correctly classified 81.82% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr50-5brain --device cuda:6 --nologging 
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr50-5brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locXYZdownr50-5brain_model_weights_2025-08-30-16-30-30.253293_bestACC-87.31.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000
[605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22]  (xydownr=500, zdownr=50)
Correctly classified 88.31% of cells
Correctly classified 91.45% of class-01 cells
Correctly classified 86.11% of class-02 cells
Correctly classified 79.59% of class-03 cells
Correctly classified 81.82% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr500-5brain --device cuda:7 --nologging 
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr500-5brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locXYZdownr500-5brain_model_weights_2025-08-30-16-31-34.664742_bestACC-88.31.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000
[605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22]  (xydownr=500, zdownr=50)
Correctly classified 88.31% of cells
Correctly classified 93.16% of class-01 cells
Correctly classified 84.26% of class-02 cells
Correctly classified 75.51% of class-03 cells
Correctly classified 81.82% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr500-5brain --device cuda:7 --nologging --lr_scheduler --focalloss
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locXYZdownr500-5brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locXYZdownr500-5brain_model_weights_2025-09-01-10-33-41.979597_bestACC-88.31.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000
[605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22]  (xydownr=200, zdownr=50, bbox_expand=50)
Correctly classified 89.80% of cells
Correctly classified 95.73% of class-01 cells
Correctly classified 85.19% of class-02 cells
Correctly classified 75.51% of class-03 cells
Correctly classified 72.73% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locXYZbbox50-5brain --device cuda:7 --nologging --lr_scheduler --focalloss
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locXYZbbox50-5brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locXYZbbox50-5brain_model_weights_2025-09-02-14-13-31.298731_bestACC-89.80.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000
[605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22]  (xydownr=500, zdownr=50)
Correctly classified 86.57% of cells
Correctly classified 85.90% of class-01 cells
Correctly classified 87.96% of class-02 cells
Correctly classified 85.71% of class-03 cells
Correctly classified 90.91% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-5brain --device cuda:6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-5brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-5brain_model_weights_2025-09-03-15-28-00.887887_bestACC-86.57.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000
[605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22] 
Correctly classified 85.32% of cells
Correctly classified 85.47% of class-01 cells
Correctly classified 87.04% of class-02 cells
Correctly classified 81.63% of class-03 cells
Correctly classified 81.82% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locLin-balsam-5brain --device cuda:5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locLin-balsam-5brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locLin-balsam-5brain_model_weights_2025-09-03-15-48-03.023560_bestACC-85.32.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1  (xydownr=500, zdownr=50)
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105]
Correctly classified 83.84% of cells
Correctly classified 89.60% of class-01 cells
Correctly classified 77.88% of class-02 cells
Correctly classified 72.13% of class-03 cells
Correctly classified 69.57% of class-04 cells
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22]
Correctly classified 93.53% of cells
Correctly classified 94.87% of class-01 cells
Correctly classified 90.74% of class-02 cells
Correctly classified 93.88% of class-03 cells
Correctly classified 90.91% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-6brain --device cuda:7
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-6brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-09-09-12-27-25.840522_bestACC-83.84.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1  (xydownr=200, zdownr=50)
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105]
Correctly classified 87.07% of cells
Correctly classified 91.61% of class-01 cells
Correctly classified 83.19% of class-02 cells
Correctly classified 78.69% of class-03 cells
Correctly classified 69.57% of class-04 cells
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22]
Correctly classified 87.31% of cells
Correctly classified 91.03% of class-01 cells
Correctly classified 85.19% of class-02 cells
Correctly classified 75.51% of class-03 cells
Correctly classified 81.82% of class-04 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locPlus-6brain --device cuda:6 --nologging 
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locPlus-6brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locPlus-6brain_model_weights_2025-09-09-12-46-31.325624_bestACC-87.07.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105]
Correctly classified 84.24% of cells
Correctly classified 85.91% of class-01 cells
Correctly classified 83.19% of class-02 cells
Correctly classified 81.97% of class-03 cells
Correctly classified 73.91% of class-04 cells
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22]
Correctly classified 88.06% of cells
Correctly classified 89.74% of class-01 cells
Correctly classified 81.48% of class-02 cells
Correctly classified 93.88% of class-03 cells
Correctly classified 90.91% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locLin-balsam-6brain --device cuda:5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locLin-balsam-6brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locLin-balsam-6brain_model_weights_2025-09-09-12-47-22.871334_bestACC-84.24.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177,  74]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177,  74]
Correctly classified 86.23% of cells
Correctly classified 91.23% of class-01 cells
Correctly classified 80.54% of class-02 cells
Correctly classified 78.12% of class-03 cells
Correctly classified 73.08% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locLin-balsam-7brain --device cuda:5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locLin-balsam-7brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locLin-balsam-7brain_model_weights_2025-09-17-00-12-12.785806_bestACC-86.23.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177,  74]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177,  74]
Correctly classified 85.89% of cells
Correctly classified 88.60% of class-01 cells
Correctly classified 83.89% of class-02 cells
Correctly classified 82.81% of class-03 cells
Correctly classified 69.23% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-7brain --device cuda:1
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-7brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-7brain_model_weights_2025-09-17-00-15-41.436482_bestACC-85.89.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177,  74], [170,   3,  69,   3]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177,  74], [170,   3,  69,   3]
Correctly classified 86.45% of cells
Correctly classified 90.14% of class-01 cells
Correctly classified 83.22% of class-02 cells
Correctly classified 86.30% of class-03 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-8brain --device cuda:2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-8brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-8brain_model_weights_2025-09-26-15-38-24.108300_bestACC-86.45.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177,  74], [170,   3,  69,   3], [128,   4, 126,   2]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177,  74], [170,   3,  69,   3], [128,   4, 126,   2]
Correctly classified 87.32% of cells
Correctly classified 86.50% of class-01 cells
Correctly classified 90.07% of class-02 cells
Correctly classified 86.81% of class-03 cells
Correctly classified 84.62% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-9brain --device cuda:2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-9brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-9brain_model_weights_2025-09-29-12-58-42.540401_bestACC-87.32.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2]
Correctly classified 87.20% of cells
Correctly classified 92.37% of class-01 cells
Correctly classified 84.46% of class-02 cells
Correctly classified 80.91% of class-03 cells
Correctly classified 61.29% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-10brain --device cuda:2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-10brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-10brain_model_weights_2025-09-30-11-56-27.957856_bestACC-87.20.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2]
Correctly classified 84.60% of cells
Correctly classified 86.65% of class-01 cells
Correctly classified 82.43% of class-02 cells
Correctly classified 87.27% of class-03 cells
Correctly classified 61.29% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.00001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-10brain --device cuda:1
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-10brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-10brain_model_weights_2025-09-30-13-37-46.300046_bestACC-84.60.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2]
Correctly classified 87.96% of cells
Correctly classified 90.19% of class-01 cells
Correctly classified 91.22% of class-02 cells
Correctly classified 84.55% of class-03 cells
Correctly classified 58.06% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-10brain --device cuda:5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-10brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-10brain_model_weights_2025-09-30-13-34-49.010633_bestACC-87.96.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4]
Correctly classified 87.46% of cells
Correctly classified 89.54% of class-01 cells
Correctly classified 88.89% of class-02 cells
Correctly classified 86.79% of class-03 cells
Correctly classified 51.85% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-11brain --device cuda:5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-11brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-11brain_model_weights_2025-09-30-17-09-28.659078_bestACC-87.46.pth

# ```
#     L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1
# train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4]
# test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4]

# ```
# python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-11brain --device cuda:5
# python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-11brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-11brain_model_weights_2025-10-01-11-53-52.123609_bestACC-87.91.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1 + L69D764P9_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7]
Correctly classified 88.24% of cells
Correctly classified 91.25% of class-01 cells
Correctly classified 85.26% of class-02 cells
Correctly classified 82.76% of class-03 cells
Correctly classified 84.00% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-12brain --device cuda:6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-12brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-12brain_model_weights_2025-10-01-12-27-30.058471_bestACC-88.24.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1 + L69D764P9_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7]
Correctly classified 87.95% of cells
Correctly classified 88.25% of class-01 cells
Correctly classified 85.26% of class-02 cells
Correctly classified 91.38% of class-03 cells
Correctly classified 84.00% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-12brain --device cuda:6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-12brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-12brain_model_weights_2025-10-01-15-41-44.657267_bestACC-87.95.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1 + L69D764P9_coloc-mask-v1 + L77D764P9_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16]
Correctly classified 87.43% of cells
Correctly classified 88.94% of class-01 cells
Correctly classified 85.03% of class-02 cells
Correctly classified 88.32% of class-03 cells
Correctly classified 76.47% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-13brain --device cuda:7
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-13brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-13brain_model_weights_2025-10-02-11-48-21.987376_bestACC-87.43.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1 + L69D764P9_coloc-mask-v1 + L77D764P9_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16]
Correctly classified 87.43% of cells
Correctly classified 90.20% of class-01 cells
Correctly classified 84.35% of class-02 cells
Correctly classified 89.05% of class-03 cells
Correctly classified 61.76% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-13brain --device cuda:6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-13brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-13brain_model_weights_2025-10-02-11-50-39.084251_bestACC-87.43.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1 + L69D764P9_coloc-mask-v1 + L77D764P9_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16]
Correctly classified 87.85% of cells
Correctly classified 88.69% of class-01 cells
Correctly classified 91.16% of class-02 cells
Correctly classified 83.21% of class-03 cells
Correctly classified 82.35% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-13brain --device cuda:5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-13brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-13brain_model_weights_2025-10-02-11-53-17.269599_bestACC-87.85.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1 + L69D764P9_coloc-mask-v1 + L77D764P9_coloc-mask-v1 + L77D764P2_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18]
Correctly classified 87.24% of cells
Correctly classified 90.75% of class-01 cells
Correctly classified 82.61% of class-02 cells
Correctly classified 84.92% of class-03 cells
Correctly classified 74.19% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-14brain --device cuda:5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-14brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-14brain_model_weights_2025-10-03-12-25-14.322755_bestACC-87.24.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1 + L69D764P9_coloc-mask-v1 + L77D764P9_coloc-mask-v1 + L77D764P2_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18]
Correctly classified 87.38% of cells
Correctly classified 91.73% of class-01 cells
Correctly classified 82.61% of class-02 cells
Correctly classified 86.51% of class-03 cells
Correctly classified 58.06% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-14brain --device cuda:6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-14brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-14brain_model_weights_2025-10-03-10-45-44.852267_bestACC-87.38.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1 + L69D764P9_coloc-mask-v1 + L77D764P9_coloc-mask-v1 + L77D764P2_coloc-mask-v1
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18]
Correctly classified 88.61% of cells
Correctly classified 93.19% of class-01 cells
Correctly classified 81.37% of class-02 cells
Correctly classified 91.27% of class-03 cells
Correctly classified 54.84% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-14brain --device cuda:7
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-14brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-14brain_model_weights_2025-10-03-10-45-50.903447_bestACC-88.61.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1 + L69D764P9_coloc-mask-v1 + L77D764P9_coloc-mask-v1 + L77D764P2_coloc-mask-v1 + L64D804P3 + L66D764P6
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18], [99, 69, 41, 10], [125, 43, 23, 7]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18], [99, 69, 41, 10], [125, 43, 23, 7]
Correctly classified 89.23% of cells
Correctly classified 91.95% of class-01 cells
Correctly classified 89.71% of class-02 cells
Correctly classified 89.52% of class-03 cells
Correctly classified 54.05% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-16brain --device cuda:1
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-16brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-16brain_model_weights_2025-10-08-18-17-36.922891_bestACC-89.23.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1 + L69D764P9_coloc-mask-v1 + L77D764P9_coloc-mask-v1 + L77D764P2_coloc-mask-v1 + L64D804P3 + L66D764P6
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18], [99, 69, 41, 10], [125, 43, 23, 7]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18], [99, 69, 41, 10], [125, 43, 23, 7]
Correctly classified 88.46% of cells
Correctly classified 91.03% of class-01 cells
Correctly classified 88.00% of class-02 cells
Correctly classified 89.52% of class-03 cells
Correctly classified 56.76% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-16brain --device cuda:2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-16brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-16brain_model_weights_2025-10-09-11-00-15.817393_bestACC-88.59.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1 + L69D764P9_coloc-mask-v1 + L77D764P9_coloc-mask-v1 + L77D764P2_coloc-mask-v1 + L64D804P3 + L66D764P6
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18], [99, 69, 41, 10], [125, 43, 23, 7]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18], [99, 69, 41, 10], [125, 43, 23, 7]
Correctly classified 88.98% of cells
Correctly classified 91.49% of class-01 cells
Correctly classified 86.29% of class-02 cells
Correctly classified 87.90% of class-03 cells
Correctly classified 75.68% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-16brain --device cuda:3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet-multiclass-bbox-locCat-balsam-16brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet-multiclass-bbox-locCat-balsam-16brain_model_weights_2025-10-09-11-39-43.754231_bestACC-88.98.pth

```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1 + L69D764P9_coloc-mask-v1 + L77D764P9_coloc-mask-v1 + L77D764P2_coloc-mask-v1 + L64D804P3 + L66D764P6
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18], [99, 69, 41, 10], [125, 43, 23, 7]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18], [99, 69, 41, 10], [125, 43, 23, 7]
Correctly classified 88.98% of cells
Correctly classified 89.43% of class-01 cells
Correctly classified 93.71% of class-02 cells
Correctly classified 89.52% of class-03 cells
Correctly classified 59.46% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain_model_weights_2025-10-08-17-59-22.439939_bestACC-88.98.pth


```
    L66D764P8 + L66D764P5 (with global loc) + L73D766P7 + L64D804P9_z300-600_z800-1000 + L64D804P3_z300-600_z800-1000 + L74D769P4_coloc-mask-v1 + L74D769P8_coloc-mask-v1 + L73D766P5_coloc-mask-v1 + L73D766P4_coloc-mask-v1 + L73D766P9_coloc-mask-v1 + L69D764P6_coloc-mask-v1 + L69D764P9_coloc-mask-v1 + L77D764P9_coloc-mask-v1 + L77D764P2_coloc-mask-v1 + L64D804P3 + L66D764P6
train: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18], [99, 69, 41, 10], [125, 43, 23, 7]
test: [605, 220,  63,  34], [554, 227,  97,  16], [438, 178, 109,  23], [473, 137,  84,   9], [462, 180,  95,  22], [385, 319, 117, 105], [430, 181, 177, 74], [170, 3, 69, 3], [128, 4, 126, 2], [142, 2, 102, 2], [139, 11, 67, 4], [108, 25, 47, 7], [86, 26, 62, 16], [29, 13, 71, 18], [99, 69, 41, 10], [125, 43, 23, 7]
Correctly classified 88.98% of cells
Correctly classified 89.43% of class-01 cells
Correctly classified 93.71% of class-02 cells
Correctly classified 89.52% of class-03 cells
Correctly classified 59.46% of class-04 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --notrain --nologging --weights_path model_weights/coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain_model_weights_2025-10-08-17-59-22.439939_bestACC-88.98.pth

python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag L35D719P5 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:1 > coloc_addbrain_trainlog/L35D719P5_addbrain-val.log
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --notrain --nologging --weights_path model_weights/L35D719P5coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain_model_weights_2025-12-16-18-51-13.162798_bestACC-87.16.pth --add_btag L35D719P5

btag=L35D719P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --notrain --nologging --weights_path model_weights/L35D719P5coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain_model_weights_2025-12-16-18-51-13.162798_bestACC-87.16.pth --add_btag ${btag}


btag=L57D855P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L57D855P5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L57D855P1
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L57D855P2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L59D878P2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L59D878P5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L64D804P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L64D804P6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L64D804P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log
######################
btag=L64D804P9
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L66D764P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L66D764P8
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L66D764P5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L66D764P6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L69D764P6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L69D764P9
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L73D766P5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L73D766P7
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L73D766P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log
##################################
btag=L73D766P9
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L74D769P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L74D769P8
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L77D764P2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L77D764P9
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L79D769P7
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L79D769P9
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L91D814P2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L91D814P6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L91D814P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L91D814P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

#########################################################################################

weights_tag=16brain-layer23
btag=L69D764P6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L69D764P9
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L73D766P5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L73D766P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L73D766P9
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:3 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L74D769P8
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L77D764P9
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L79D769P7
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L79D769P9
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:3 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L91D814P2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L91D814P6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L91D814P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

##################################################################################################################################################################################################################################################################################################################################################


btag=L69D764P6
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L69D764P9
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L73D766P5
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L73D766P4
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer
####################
btag=L73D766P9
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L74D769P8
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L77D764P9
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L79D769P7
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L79D769P9
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L91D814P2
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L91D814P6
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L91D814P3
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

#########################################################################################

weights_tag=16brain-layer23-bgrm

btag=L64D804P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L64D804P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L64D804P6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L64D804P9
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L66D764P6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L66D764P8
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L74D769P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L73D766P7
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

weights_tag=16brain-layer23-bgrm

btag=L64D804P3
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 1 --zmax 100 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 100 --zmax 200 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 200 --zmax 300 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 400 --zmax 500 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 500 --zmax 600 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 600 --zmax 700 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 700 --zmax 800 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 800 --zmax 900 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L64D804P4
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 1 --zmax 100 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 100 --zmax 200 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 700 --zmax 800 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 800 --zmax 900 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L64D804P6
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 1 --zmax 100 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 100 --zmax 200 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 200 --zmax 300 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 400 --zmax 500 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 700 --zmax 800 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 800 --zmax 900 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L64D804P9
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 1 --zmax 100 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 100 --zmax 200 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 400 --zmax 500 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 500 --zmax 600 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 600 --zmax 700 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 700 --zmax 800 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 800 --zmax 900 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L66D764P6
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 1 --zmax 100 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 100 --zmax 200 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 200 --zmax 300 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 300 --zmax 400 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 400 --zmax 500 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 500 --zmax 600 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 600 --zmax 700 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 700 --zmax 800 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 800 --zmax 900 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L66D764P8
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L74D769P4
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 900 --zmax -1 --batch_size 512 # hummer


weights_tag=16brain-layer23-bgrm
## Acc too low
btag=L35D719P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 512 --learning_rate 0.001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L57D855P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0005 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log
#####
### Inferring
btag=L35D719P5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:0 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L57D855P2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L73D766P7
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L91D814P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L66D764P5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:3 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L66D764P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:3 > coloc_addbrain_trainlog/${btag}_addbrain-val.log
### Inferring
btag=L59D878P2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L59D878P5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L77D764P2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}_addbrain-val.log
### Training layer2/3 2nd round
weights_tag=16brain-layer23-bgrm-round2
btag=L69D764P6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:1 > coloc_addbrain_trainlog/${btag}-${weights_tag}_addbrain-val.log

btag=L69D764P9
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:3 > coloc_addbrain_trainlog/${btag}-${weights_tag}_addbrain-val.log

btag=L73D766P5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:4 > coloc_addbrain_trainlog/${btag}-${weights_tag}_addbrain-val.log
### Error
btag=L73D766P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p4-resnet50-multiclass-bbox-locCat-balsam-${weights_tag} --device cuda:5 > coloc_addbrain_trainlog/${btag}-${weights_tag}_addbrain-val.log

#########

cd /cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph
conda activate pyg24


#######

weights_tag=16brain-layer23-bgrm
btag=L35D719P5
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 1 --zmax 100 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 100 --zmax 200 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 200 --zmax 300 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 300 --zmax 400 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 400 --zmax 500 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 500 --zmax 600 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 600 --zmax 700 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 700 --zmax 800 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 800 --zmax 900 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L57D855P2
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 1 --zmax 100 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 100 --zmax 200 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 200 --zmax 300 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 300 --zmax 400 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 400 --zmax 500 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 500 --zmax 550 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 550 --zmax 600 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 600 --zmax 650 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 650 --zmax 700 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 700 --zmax 800 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L73D766P7
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 1 --zmax 100 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 100 --zmax 200 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 200 --zmax 300 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 300 --zmax 400 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 400 --zmax 500 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 500 --zmax 600 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 600 --zmax 700 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 700 --zmax 800 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 800 --zmax 900 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L91D814P4
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 1 --zmax 100 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 100 --zmax 200 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 200 --zmax 300 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 300 --zmax 400 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 400 --zmax 500 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 500 --zmax 600 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 600 --zmax 700 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 700 --zmax 800 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 800 --zmax 900 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L66D764P5
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 1 --zmax 100 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 100 --zmax 200 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 200 --zmax 300 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 300 --zmax 400 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 400 --zmax 500 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 500 --zmax 600 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 600 --zmax 700 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 700 --zmax 750 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 750 --zmax 800 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 800 --zmax 900 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L66D764P3
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 1 --zmax 100 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 100 --zmax 200 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 200 --zmax 300 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 300 --zmax 400 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 400 --zmax 500 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 500 --zmax 600 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:4 --zmin 600 --zmax 700 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 700 --zmax 800 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 800 --zmax 900 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 900 --zmax -1 --batch_size 512 # hummer

###########################
weights_tag=16brain-layer23-bgrm

btag=L59D878P2
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 1 --zmax 100 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 100 --zmax 200 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 200 --zmax 300 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 300 --zmax 400 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 400 --zmax 500 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 600 --zmax 700 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 700 --zmax 800 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 800 --zmax 900 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L59D878P5
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 1 --zmax 100 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 100 --zmax 200 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 200 --zmax 300 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 300 --zmax 400 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 400 --zmax 500 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 500 --zmax 550 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 550 --zmax 600 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 600 --zmax 650 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:2 --zmin 650 --zmax 700 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 700 --zmax 800 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 800 --zmax 900 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:1 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L77D764P2
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 1 --zmax 100 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 100 --zmax 200 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 200 --zmax 300 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 300 --zmax 400 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 400 --zmax 500 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 500 --zmax 600 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 600 --zmax 700 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 700 --zmax 800 --batch_size 512 # yukon
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 800 --zmax 900 --batch_size 512 # hummer
# python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --weights_tag ${weights_tag} --gtag P4 --device cuda:3 --zmin 900 --zmax -1 --batch_size 512 # hummer
##################################################################################################################################################################################################################################################################################################################################################



```
Infer a brain
```
##################################################################################################################################################################################################################################################################################################################################################
btag=L57D855P4
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L57D855P5
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L57D855P1
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L57D855P2
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L59D878P2
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L59D878P5
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L64D804P4
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L64D804P6
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L64D804P3
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L64D804P9
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L66D764P3
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L66D764P8
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L66D764P5
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L66D764P6
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L69D764P6
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L69D764P9
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L73D766P5
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L73D766P7
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L73D766P4
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L73D766P9
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L74D769P4
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L74D769P8
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L77D764P2
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L77D764P9
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L79D769P7
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L79D769P9
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L91D814P2
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L91D814P6
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L91D814P3
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L91D814P4
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --btag ${btag} --gtag P4 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer


#####################################################

```
testcoloc_L106P3
tensor([  0, 301,  95, 527,   3])
Correctly classified 91.30% of cells
Correctly classified 96.15% of class-01 cells
Correctly classified 62.50% of class-02 cells
Correctly classified 93.10% of class-03 cells
```
python coloc_classifier_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet --lr_scheduler --focalloss

```
testcoloc_L106P3
L106P3_chAlign
tensor([  0, 301,  95, 527,   3])
tensor([  0, 294, 150, 378,  22])
Correctly classified 84.18% of cells
Correctly classified 79.03% of class-01 cells
Correctly classified 74.19% of class-02 cells
Correctly classified 92.77% of class-03 cells
```
python coloc_classifier_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet --lr_scheduler --focalloss
```
L106P3_chAlign
tensor([  0, 294, 150, 378,  22])
Correctly classified 89.29% of cells
Correctly classified 94.12% of class-01 cells
Correctly classified 75.00% of class-02 cells
Correctly classified 93.94% of class-03 cells
```
python coloc_classifier_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet --lr_scheduler --focalloss


```
testcoloc_L106P3
L106P3_chAlign
Correctly classified 76.27% of cells
Correctly classified 67.74% of class-01 cells
Correctly classified 58.06% of class-02 cells
Correctly classified 90.36% of class-03 cells
```
python coloc_classifierBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet --lr_scheduler --focalloss

```
L106P3_chAlign
Correctly classified 88.10% of cells
Correctly classified 88.24% of class-01 cells
Correctly classified 75.00% of class-02 cells
Correctly classified 96.97% of class-03 cells
```
python coloc_classifierBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet --lr_scheduler --focalloss


```
testcoloc_L106P3
L106P3_chAlign
Correctly classified 85.31% of cells
Correctly classified 77.42% of class-01 cells
Correctly classified 80.65% of class-02 cells
Correctly classified 93.98% of class-03 cells
```
python coloc_classifier_multiclass_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass --lr_scheduler --focalloss --device cuda:1




```
L106P3_chAlign
Correctly classified 89.29% of cells
Correctly classified 97.06% of class-01 cells
Correctly classified 68.75% of class-02 cells
Correctly classified 93.94% of class-03 cells
```
python coloc_classifier_multiclass_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass --lr_scheduler --focalloss --device cuda:1




```
L106P3_chAlign
Correctly classified 90.48% of cells
Correctly classified 97.06% of class-01 cells
Correctly classified 68.75% of class-02 cells
Correctly classified 96.97% of class-03 cells
```
python coloc_classifier_multiclassBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox --lr_scheduler --focalloss --device cuda:1 --nologging
python coloc_classifier_multiclassBbox_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox_model_weights_2025-07-16-12-17-10.906495_bestACC-90.48.pth

```
L106P5 + L106P3
[252, 137, 434], [294, 150, 378]
Correctly classified 88.17% of cells
Correctly classified 92.86% of class-01 cells
Correctly classified 67.86% of class-02 cells
Correctly classified 93.90% of class-03 cells
```
python coloc_classifier_multiclassBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-2brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifier_multiclassBbox_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-2brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-2brain_model_weights_2025-08-07-15-54-23.528101_bestACC-88.17.pth


```
L106P5 + L106P3
[252, 137, 434], [294, 150, 378]
Correctly classified 89.35% of cells
Correctly classified 98.21% of class-01 cells
Correctly classified 64.29% of class-02 cells
Correctly classified 93.90% of class-03 cells
```
python coloc_classifier_multiclass_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-2brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifier_multiclass_trainval.py --proj_name coloc-p14-resnet-multiclass-2brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-2brain_model_weights_2025-08-07-16-22-17.756718_bestACC-89.35.pth


```
L106P5 + L106P3
[252, 137, 434], [294, 150, 378] = [546, 287, 812]
Correctly classified 84.02% of cells
Correctly classified 82.14% of class-01 cells
Correctly classified 67.86% of class-02 cells
Correctly classified 93.90% of class-03 cells
```
python coloc_classifier_trainval.py --epochs 100 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-2brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifier_trainval.py --proj_name coloc-p14-resnet-2brain --notrain --nologging --device cuda:1 --weights_path model_weights/coloc-p14-resnet-2brain_model_weights_2025-08-17-18-13-16.752883_bestACC-84.02.pth


```
L106P5 + L106P3
[252, 137, 434], [294, 150, 378] = [546, 287, 812]
Correctly classified 87.57% of cells
Correctly classified 85.71% of class-01 cells
Correctly classified 85.71% of class-02 cells
Correctly classified 91.46% of class-03 cells
```
python coloc_classifierBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-2brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifierBbox_trainval.py --proj_name coloc-p14-resnet-bbox-2brain --notrain --nologging --device cuda:1 --weights_path model_weights/coloc-p14-resnet-bbox-2brain_model_weights_2025-08-08-16-04-21.447864_bestACC-87.57.pth



```
L106P5 + L106P3 + 102P1
[252, 137, 434], [294, 150, 378], [194,  76, 442]
Correctly classified 86.78% of cells
Correctly classified 89.04% of class-01 cells
Correctly classified 83.78% of class-02 cells
Correctly classified 90.24% of class-03 cells
```
python coloc_classifierBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-3brain --lr_scheduler --focalloss --device cuda:1 --nologging
python coloc_classifierBbox_trainval.py --proj_name coloc-p14-resnet-bbox-3brain --notrain --nologging --device cuda:1 --weights_path model_weights/coloc-p14-resnet-bbox-3brain_model_weights_2025-08-19-15-21-18.024215_bestACC-86.78.pth


```
L106P5 + L106P3 + 102P1
[252, 137, 434], [294, 150, 378], [194,  76, 442]
Correctly classified 85.12% of cells
Correctly classified 83.56% of class-01 cells
Correctly classified 75.68% of class-02 cells
Correctly classified 93.50% of class-03 cells
```
python coloc_classifier_multiclassBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-3brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifier_multiclassBbox_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-3brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-3brain_model_weights_2025-08-26-10-28-37.408275_bestACC-85.12.pth

```
L106P5 + L106P3 + 102P1
[252, 137, 434], [294, 150, 378], [194,  76, 442]
Correctly classified 80.99% of cells
Correctly classified 80.82% of class-01 cells
Correctly classified 62.16% of class-02 cells
Correctly classified 92.68% of class-03 cells
```
python coloc_classifierBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-loc-3brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifierBboxLoc_trainval.py --proj_name coloc-p14-resnet-bbox-loc-3brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-bbox-loc-3brain_model_weights_2025-08-26-11-38-29.291392_bestACC-80.99.pth


```
L106P5 + L106P3 + 102P1
[252, 137, 434], [294, 150, 378], [194,  76, 442]
Correctly classified 85.54% of cells
Correctly classified 87.67% of class-01 cells
Correctly classified 70.27% of class-02 cells
Correctly classified 91.87% of class-03 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-loc-3brain --lr_scheduler --focalloss --device cuda:6 --nologging
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-loc-3brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-loc-3brain_model_weights_2025-08-26-11-41-07.261762_bestACC-85.54.pth


```
L106P5 + L106P3 + 102P1 (downr = 500)
[252, 137, 434], [294, 150, 378], [194,  76, 442]
Correctly classified 84.30% of cells
Correctly classified 87.67% of class-01 cells
Correctly classified 78.38% of class-02 cells
Correctly classified 88.62% of class-03 cells
```
python coloc_classifierBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-loc-3brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifierBboxLoc_trainval.py --proj_name coloc-p14-resnet-bbox-loc-3brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-bbox-loc-3brain_model_weights_2025-08-26-12-11-32.028797_bestACC-84.30.pth

```
L106P5 + L106P3 + 102P1 (downr = 500)
[252, 137, 434], [294, 150, 378], [194,  76, 442]
Correctly classified 88.02% of cells
Correctly classified 87.67% of class-01 cells
Correctly classified 83.78% of class-02 cells
Correctly classified 92.68% of class-03 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-loc-3brain --lr_scheduler --focalloss --device cuda:6 --nologging
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-loc-3brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-loc-3brain_model_weights_2025-08-26-12-15-30.825434_bestACC-88.02.pth

```
L106P5 + L106P3 + 102P1 (downr = 500, 50)
[252, 137, 434], [294, 150, 378], [194,  76, 442]
Correctly classified 88.02% of cells
Correctly classified 90.41% of class-01 cells
Correctly classified 81.08% of class-02 cells
Correctly classified 91.87% of class-03 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locXYZ-3brain --lr_scheduler --focalloss --device cuda:6 --nologging
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-locXYZ-3brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-locXYZ-3brain_model_weights_2025-08-26-13-06-55.552126_bestACC-88.02.pth

```
L106P5 + L106P3 + 102P1 (downr = 500, 100)
[252, 137, 434], [294, 150, 378], [194,  76, 442]
Correctly classified 86.36% of cells
Correctly classified 89.04% of class-01 cells
Correctly classified 81.08% of class-02 cells
Correctly classified 90.24% of class-03 cells
```
python coloc_classifierBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-locXYZ-3brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifierBboxLoc_trainval.py --proj_name coloc-p14-resnet-bbox-locXYZ-3brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-bbox-locXYZ-3brain_model_weights_2025-08-26-13-03-17.314973_bestACC-86.78.pth

```
L106P5 + L106P3 + 102P1 
[252, 137, 434], [294, 150, 378], [194,  76, 442]
Correctly classified 87.60% of cells
Correctly classified 87.67% of class-01 cells
Correctly classified 81.08% of class-02 cells
Correctly classified 93.50% of class-03 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locLin-3brain --lr_scheduler --focalloss --device cuda:6 --nologging
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-locLin-3brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-locLin-3brain_model_weights_2025-08-26-14-18-54.561632_bestACC-87.60.pth

```
L106P5 + L106P3 + 102P1 
[252, 137, 434], [294, 150, 378], [194,  76, 442]
Correctly classified 86.78% of cells
Correctly classified 87.67% of class-01 cells
Correctly classified 75.68% of class-02 cells
Correctly classified 91.87% of class-03 cells
```
python coloc_classifierBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-locLin-3brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifierBboxLoc_trainval.py --proj_name coloc-p14-resnet-bbox-locLin-3brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-bbox-locLin-3brain_model_weights_2025-08-26-14-15-15.741039_bestACC-86.78.pth


```
L106P5 + L106P3 + L102P1 + L102P2
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ]
Correctly classified 92.02% of cells
Correctly classified 91.67% of class-01 cells
Correctly classified 79.41% of class-02 cells
Correctly classified 94.63% of class-03 cells
```
python coloc_classifier_multiclassBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-4brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifier_multiclassBbox_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-4brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-4brain_model_weights_2025-08-27-10-19-43.148038_bestACC-92.02.pth

```
L106P5 + L106P3 + L102P1 + L102P2
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ]
Correctly classified 92.78% of cells
Correctly classified 95.83% of class-01 cells
Correctly classified 88.24% of class-02 cells
Correctly classified 94.63% of class-03 cells
```
python coloc_classifierBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-4brain --lr_scheduler --focalloss --device cuda:6 --nologging
python coloc_classifierBbox_trainval.py --proj_name coloc-p14-resnet-bbox-4brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-bbox-4brain_model_weights_2025-08-27-10-15-09.300047_bestACC-92.78.pth

```
L106P5 + L106P3 + L102P1 + L102P2 (xydownr=500, zdownr=50)
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ]
Correctly classified 92.78% of cells
Correctly classified 95.83% of class-01 cells
Correctly classified 73.53% of class-02 cells
Correctly classified 97.32% of class-03 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locXYZ-4brain --lr_scheduler --focalloss --device cuda:5 --nologging
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-locXYZ-4brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-locXYZ-4brain_model_weights_2025-08-27-11-06-13.662683_bestACC-92.78.pth

```
L106P5 + L106P3 + L102P1 + L102P2 (xydownr=500, zdownr=50)
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ]
Correctly classified 92.40% of cells
Correctly classified 90.28% of class-01 cells
Correctly classified 85.29% of class-02 cells
Correctly classified 97.32% of class-03 cells
```
python coloc_classifierBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-locXYZ-4brain --lr_scheduler --focalloss --device cuda:4 --nologging
python coloc_classifierBboxLoc_trainval.py --proj_name coloc-p14-resnet-bbox-locXYZ-4brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-bbox-locXYZ-4brain_model_weights_2025-08-27-11-01-48.886803_bestACC-92.40.pth

```
L106P5 + L106P3 + L102P1 + L102P2 (xydownr=500, zdownr=150)
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ]
Correctly classified 92.78% of cells
Correctly classified 95.83% of class-01 cells
Correctly classified 70.59% of class-02 cells
Correctly classified 97.32% of class-03 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locXYZ-4brain --lr_scheduler --focalloss --device cuda:5 --nologging
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-locXYZ-4brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-locXYZ-4brain_model_weights_2025-08-27-11-30-40.440687_bestACC-92.78.pth

```
L106P5 + L106P3 + L102P1 + L102P2 (xydownr=500, zdownr=150)
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ]
Correctly classified 91.63% of cells
Correctly classified 90.28% of class-01 cells
Correctly classified 79.41% of class-02 cells
Correctly classified 95.97% of class-03 cells
```
python coloc_classifierBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-locXYZ-4brain --lr_scheduler --focalloss --device cuda:4 --nologging
python coloc_classifierBboxLoc_trainval.py --proj_name coloc-p14-resnet-bbox-locXYZ-4brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-bbox-locXYZ-4brain_model_weights_2025-08-27-11-26-49.815210_bestACC-91.63.pth

```
L106P5 + L106P3 + L102P1 + L102P2 (xydownr=500, zdownr=20)
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ]
Correctly classified 91.25% of cells
Correctly classified 94.44% of class-01 cells
Correctly classified 67.65% of class-02 cells
Correctly classified 96.64% of class-03 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locXYZz20-4brain --lr_scheduler --focalloss --device cuda:6 --nologging
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-locXYZz20-4brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-locXYZz20-4brain_model_weights_2025-08-27-11-34-05.808508_bestACC-91.25.pth

```
L106P5 + L106P3 + L102P1 + L102P2 (xydownr=500, zdownr=20)
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ]
Correctly classified 90.87% of cells
Correctly classified 91.67% of class-01 cells
Correctly classified 76.47% of class-02 cells
Correctly classified 96.64% of class-03 cells
```
python coloc_classifierBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-locXYZz20-4brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifierBboxLoc_trainval.py --proj_name coloc-p14-resnet-bbox-locXYZz20-4brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-bbox-locXYZz20-4brain_model_weights_2025-08-27-11-30-48.169746_bestACC-90.87.pth


```
L106P5 + L106P3 + L102P1 + L102P2 (xydownr=500, zdownr=50)
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ]
Correctly classified 92.40% of cells
Correctly classified 93.06% of class-01 cells
Correctly classified 82.35% of class-02 cells
Correctly classified 95.30% of class-03 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-4brain --lr_scheduler --focalloss --device cuda:3 --nologging
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-locCat-4brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-locCat-4brain_model_weights_2025-08-27-11-45-10.617793_bestACC-92.40.pth

```
L106P5 + L106P3 + L102P1 + L102P2 (xydownr=500, zdownr=20)
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ]
Correctly classified 92.02% of cells
Correctly classified 93.06% of class-01 cells
Correctly classified 82.35% of class-02 cells
Correctly classified 96.64% of class-03 cells
```
python coloc_classifierBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-locCat-4brain --lr_scheduler --focalloss --device cuda:2 --nologging
python coloc_classifierBboxLoc_trainval.py --proj_name coloc-p14-resnet-bbox-locCat-4brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-bbox-locCat-4brain_model_weights_2025-08-27-11-40-18.801794_bestACC-92.02.pth


```
L106P5 + L106P3 + L102P1 + L102P2
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ]
Correctly classified 92.78% of cells
Correctly classified 95.83% of class-01 cells
Correctly classified 73.53% of class-02 cells
Correctly classified 96.64% of class-03 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locLin-4brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-locLin-4brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-locLin-4brain_model_weights_2025-08-27-11-57-10.945241_bestACC-92.78.pth

```
L106P5 + L106P3 + L102P1 + L102P2
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ]
Correctly classified 93.16% of cells
Correctly classified 95.83% of class-01 cells
Correctly classified 82.35% of class-02 cells
Correctly classified 96.64% of class-03 cells
```
python coloc_classifierBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-locLin-4brain --lr_scheduler --focalloss --device cuda:6 --nologging
python coloc_classifierBboxLoc_trainval.py --proj_name coloc-p14-resnet-bbox-locLin-4brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-bbox-locLin-4brain_model_weights_2025-08-27-11-53-18.782176_bestACC-93.16.pth


```
L106P5 + L106P3 + L102P1 + L102P2 + L102P1_mask_v1
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92], [174,  43, 599]
Correctly classified 89.05% of cells
Correctly classified 90.36% of class-01 cells
Correctly classified 78.05% of class-02 cells
Correctly classified 93.81% of class-03 cells
```
python coloc_classifierBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-5brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifierBbox_trainval.py --proj_name coloc-p14-resnet-bbox-5brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-bbox-5brain_model_weights_2025-09-03-13-27-52.803334_bestACC-89.05.pth

```
L106P5 + L106P3 + L102P1 + L102P2 + L102P1_mask_v1
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599]
Correctly classified 90.78% of cells
Correctly classified 92.77% of class-01 cells
Correctly classified 75.61% of class-02 cells
Correctly classified 95.71% of class-03 cells
```
python coloc_classifier_multiclassBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-5brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifier_multiclassBbox_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-5brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-5brain_model_weights_2025-09-03-13-57-18.783819_bestACC-90.78.pth

```
L106P5 + L106P3 + L102P1 + L102P2 + L102P1_mask_v1
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599]
Correctly classified 90.20% of cells
Correctly classified 91.57% of class-01 cells
Correctly classified 68.29% of class-02 cells
Correctly classified 96.67% of class-03 cells
```
python coloc_classifierBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-locLin-5brain --lr_scheduler --focalloss --device cuda:6 --nologging
python coloc_classifierBboxLoc_trainval.py --proj_name coloc-p14-resnet-bbox-locLin-5brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-bbox-locLin-5brain_model_weights_2025-09-03-13-57-04.720855_bestACC-90.20.pth

```
L106P5 + L106P3 + L102P1 + L102P2 + L102P1_mask_v1 (xydownr=500, zdownr=50)
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599]
Correctly classified 91.07% of cells
Correctly classified 91.57% of class-01 cells
Correctly classified 78.05% of class-02 cells
Correctly classified 95.71% of class-03 cells
```
python coloc_classifier_multiclassBboxLoc_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-5brain --lr_scheduler --focalloss --device cuda:5 --nologging
python coloc_classifier_multiclassBboxLoc_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-locCat-5brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-locCat-5brain_model_weights_2025-09-03-14-03-34.805732_bestACC-91.07.pth

```
L106P5 + L106P3 + L102P1 + L102P2 + L102P1_mask_v1 (xydownr=500, zdownr=50)
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599]
Correctly classified 91.35% of cells
Correctly classified 93.98% of class-01 cells
Correctly classified 85.37% of class-02 cells
Correctly classified 94.29% of class-03 cells
```
python coloc_classifier_multiclassBboxBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-balsam-5brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifier_multiclassBboxBalsam_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-balsam-5brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-balsam-5brain_model_weights_2025-09-03-14-49-19.411036_bestACC-91.35.pth

```
L106P5 + L106P3 + L102P1 + L102P2 + L102P1_mask_v1 (xydownr=500, zdownr=50)
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599]
Correctly classified 89.34% of cells
Correctly classified 87.95% of class-01 cells
Correctly classified 70.73% of class-02 cells
Correctly classified 95.24% of class-03 cells
```
python coloc_classifierBboxBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-bbox-balsam-5brain --lr_scheduler --focalloss --device cuda:6
python coloc_classifierBboxBalsam_trainval.py --proj_name coloc-p14-resnet-bbox-balsam-5brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-bbox-balsam-5brain_model_weights_2025-09-03-14-49-42.086722_bestACC-89.34.pth

```
L106P5 + L106P3 + L102P1 + L102P2 + L102P1_mask_v1 (xydownr=500, zdownr=50)
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599]
Correctly classified 90.78% of cells
Correctly classified 91.57% of class-01 cells
Correctly classified 85.37% of class-02 cells
Correctly classified 93.33% of class-03 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-5brain --lr_scheduler --focalloss --device cuda:7
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-5brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-locCat-balsam-5brain_model_weights_2025-09-03-15-21-55.078421_bestACC-90.78.pth

```
L106P5 + L106P3 + L102P1 + L102P2 + L102P1_mask_v1
[252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599]
Correctly classified 89.91% of cells
Correctly classified 92.77% of class-01 cells
Correctly classified 75.61% of class-02 cells
Correctly classified 92.86% of class-03 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locLin-balsam-5brain --lr_scheduler --focalloss --device cuda:4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-locLin-balsam-5brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-locLin-balsam-5brain_model_weights_2025-09-03-15-45-03.105036_bestACC-89.91.pth



```
L106P5 + L106P3 + L102P1 + L102P2 + L102P1_mask_v1 + L102P1_mask_v2
train: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599], [272, 144, 366]
test: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599], [272, 144, 366]
Correctly classified 85.95% of cells
Correctly classified 84.78% of class-01 cells
Correctly classified 76.47% of class-02 cells
Correctly classified 89.78% of class-03 cells
train: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599], [272, 144, 366]
test: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599]
Correctly classified 89.34% of cells
Correctly classified 91.57% of class-01 cells
Correctly classified 78.05% of class-02 cells
Correctly classified 90.95% of class-03 cells
train: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599], [272, 144, 366]
test: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ]
Correctly classified 90.87% of cells
Correctly classified 91.67% of class-01 cells
Correctly classified 73.53% of class-02 cells
Correctly classified 93.96% of class-03 cells
```
python coloc_classifier_multiclassBboxBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-balsam-6brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifier_multiclassBboxBalsam_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-balsam-6brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-balsam-6brain_model_weights_2025-09-09-10-42-39.503912_bestACC-85.95.pth


```
L102P1_mask_v1 + L102P1_mask_v2
[174,  43, 599], [272, 144, 366]
Correctly classified 87.12% of cells
Correctly classified 93.94% of class-01 cells
Correctly classified 64.71% of class-02 cells
Correctly classified 88.50% of class-03 cells
```
python coloc_classifier_multiclassBboxBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-balsam-2maskbrain --lr_scheduler --focalloss --device cuda:5 --nologging
python coloc_classifier_multiclassBboxBalsam_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-balsam-2maskbrain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-balsam-2maskbrain_model_weights_2025-09-09-10-58-48.501783_bestACC-87.12.pth


```
L106P5 + L106P3 + L102P1 + L102P2 + L102P1_mask_v1 + L102P1_mask_v2
train: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599], [272, 144, 366]
test: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599], [272, 144, 366]
Correctly classified 87.35% of cells
Correctly classified 89.13% of class-01 cells
Correctly classified 78.43% of class-02 cells
Correctly classified 90.67% of class-03 cells
train: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599], [272, 144, 366]
test: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599]
Correctly classified 87.90% of cells
Correctly classified 92.77% of class-01 cells
Correctly classified 68.29% of class-02 cells
Correctly classified 93.33% of class-03 cells
```
python coloc_classifier_multiclassBbox_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-6brain --lr_scheduler --focalloss --device cuda:7 --nologging
python coloc_classifier_multiclassBbox_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-6brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-6brain_model_weights_2025-09-09-11-15-29.716255_bestACC-87.35.pth


```
L106P5 + L106P3 + L102P1 + L102P2 + L102P1_mask_v1 + L102P1_mask_v2 (xydownr=200, zdownr=50)
train: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599], [272, 144, 366]
test: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599], [272, 144, 366]
Correctly classified 86.18% of cells
Correctly classified 86.23% of class-01 cells
Correctly classified 78.43% of class-02 cells
Correctly classified 89.78% of class-03 cells
train: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599], [272, 144, 366]
test: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599]
Correctly classified 92.80% of cells
Correctly classified 93.98% of class-01 cells
Correctly classified 87.80% of class-02 cells
Correctly classified 93.81% of class-03 cells
train: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599], [272, 144, 366]
test: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ]
Correctly classified 91.25% of cells
Correctly classified 93.06% of class-01 cells
Correctly classified 79.41% of class-02 cells
Correctly classified 92.62% of class-03 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-09-09-10-47-30.797068_bestACC-86.18.pth


```
L106P5 + L106P3 + L102P1 + L102P2 + L102P1_mask_v1 + L102P1_mask_v2 (xydownr=500, zdownr=50)
train: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599], [133,  87, 296]
test: [252, 137, 434], [294, 150, 378], [194,  76, 442], [84, 24, 92 ], [174,  43, 599], [133,  87, 296]
Correctly classified 92.75% of cells
Correctly classified 91.84% of class-01 cells
Correctly classified 94.23% of class-02 cells
Correctly classified 95.40% of class-03 cells
```
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:6
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --notrain --nologging --weights_path model_weights/coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-09-09-15-12-01.487732_bestACC-92.75.pth

############################ Done
btag=L82D711P2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:2 > coloc_addbrain_trainlog/${btag}_addbrain-val.log
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --gtag P14 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --notrain --nologging --weights_path model_weights/L35D719P5coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain_model_weights_2025-12-16-18-51-13.162798_bestACC-87.16.pth --add_btag ${btag}

btag=L82D711P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --gtag P14 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --notrain --nologging --weights_path model_weights/L35D719P5coloc-p4-resnet50-multiclass-bbox-locCat-balsam-16brain_model_weights_2025-12-16-18-51-13.162798_bestACC-87.16.pth --add_btag ${btag}

btag=L87P1
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L88P2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L92P2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log
################################# Cant make patches
btag=L88P1
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L88P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L92P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L94P1
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L95P2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L96P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log
############################## No ann
btag=L86P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:5 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L87D868P2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log
######################### No infer
btag=L88P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L90P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L92P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L94P3sox9toproneun
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L94P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L95P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L95P4
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L97P1
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L97P2
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L106P3
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log

btag=L106P5
python coloc_classifier_multiclassBboxLocBalsam_trainval.py --add_btag ${btag} --gtag P14 --epochs 500 --batch_size 128 --learning_rate 0.0001 --proj_name coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain --lr_scheduler --focalloss --device cuda:4 > coloc_addbrain_trainlog/${btag}_addbrain-val.log


btag=L82D711P2
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P2coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-52-42.217806_bestACC-90.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P2coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-52-42.217806_bestACC-90.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P2coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-52-42.217806_bestACC-90.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P2coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-52-42.217806_bestACC-90.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P2coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-52-42.217806_bestACC-90.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P2coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-52-42.217806_bestACC-90.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P2coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-52-42.217806_bestACC-90.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P2coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-52-42.217806_bestACC-90.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P2coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-52-42.217806_bestACC-90.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P2coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-52-42.217806_bestACC-90.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L82D711P3
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P3coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-40-55.911337_bestACC-100.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P3coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-40-55.911337_bestACC-100.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P3coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-40-55.911337_bestACC-100.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P3coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-40-55.911337_bestACC-100.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:0 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P3coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-40-55.911337_bestACC-100.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P3coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-40-55.911337_bestACC-100.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P3coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-40-55.911337_bestACC-100.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:0 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P3coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-40-55.911337_bestACC-100.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:0 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P3coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-40-55.911337_bestACC-100.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --weights_path model_weights/L82D711P3coloc-p14-resnet-multiclass-bbox-locCat-balsam-6brain_model_weights_2025-12-20-10-40-55.911337_bestACC-100.00.pth --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L87P1
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer


btag=L88P2
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L92P2
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

################################# Cant make patches

btag=L88P1
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L88P3
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L92P4
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L94P1
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L95P2
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L96P3
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

################################### No ann
btag=L86P4
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L87D868P2
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag male --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

############################## No train

btag=L88P4
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L90P3
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L92P3
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L94P3sox9toproneun
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L94P4
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:1 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:1 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L95P3
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:5 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L95P4
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L97P1
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L97P2
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag extrabrains --btag ${btag} --gtag P14 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L106P3
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer

btag=L106P5
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 1 --zmax 100 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 100 --zmax 200 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 200 --zmax 300 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 300 --zmax 400 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 400 --zmax 500 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 500 --zmax 600 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 600 --zmax 700 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 700 --zmax 800 --batch_size 512 # yukon
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 800 --zmax 900 --batch_size 512 # hummer
python coloc_classifier-multiclassBboxLoc_inferNIS.py --ptag female --btag ${btag} --gtag P14 --device cuda:4 --zmin 900 --zmax -1 --batch_size 512 # hummer

#######################################