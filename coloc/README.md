# NIS guided co-localization

## Inference of whole-brain co-localization
1. Download model weights from G-drive [URL](https://drive.google.com/drive/folders/1nrz8TaLUStn23Q6YzgocjwCyDk9_vKWK?usp=sharing). Put models under `model_weights`
2. Use proper parameters for `python coloc_classifier-multiclassBboxLoc_inferNIS.py`, see `coloc_trainval_infer_cmd.sh` for usage examples.
3. Classification labels will be saved into `saver` in `coloc_classifier-multiclassBboxLoc_inferNIS.py`, 

## Train your own models
1. Generate training patches after NIS is done. 
    - Replace paths to NIS results in `generate_nis_patch_multichannel.py`.
    - Generate masks first if you want patches from specific regions, e.g., using function `generate_mask_layer23()` in `generate_nis_patch_multichannel.py`.
    - Use proper parameters for `python generate_nis_patch_multichannel.py`, see `generate_coloc_ann_patch.sh` for usage examples.
2. Manual annotation
    - Replace patch_with_bboxes path in `annotation_gui_cmd.sh`, 
    - Run the matlab commands in `annotation_gui_cmd.sh`. Follow the instruction of manual annotation to complete the annotation.
3. Train the model
    - Use proper parameters for `python coloc_classifier_multiclassBboxLocBalsam_trainval.py`,  see `coloc_trainval_infer_cmd.sh` for usage examples.