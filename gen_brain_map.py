from brainmap_nogui import Brainmap
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate downsampled whole brain map from NIS results')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--stitch-root', type=str, default='downloads/stitch_tform', help='Dir to save undoubled NIS ID')
    parser.add_argument('--map-type', type=str, default='cell count', choices=['avg volume', 'cell count'], help='Type of whole brain map')
    parser.add_argument('--overlap-ratio', type=float, default=0.2, help='Overlapping ratio used for stitching')
    parser.add_argument('--nis-output-root', type=str, default='downloads/cpp_output', help='Output dir of NIS cpp')
    parser.add_argument('--num-column', type=int, default=2, help='Number of columns of tiles')
    parser.add_argument('--num-row', type=int, default=2, help='Number of rows of tiles')
    parser.add_argument('--auto-stitch-root', type=str, default='downloads/stitch_tform')
    parser.add_argument('--manual-stitch-root', type=str, default='downloads/imaris_stitch_tform')
    parser.add_argument('--temp-root', type=str, default='downloads/tmp', help='Dir to save undoubled NIS ID')
    parser.add_argument('--channel-layer', type=str, default=None, help='Apply a channel layer after cell_coloc/coloc.py')

    args = parser.parse_args()
    os.makedirs(args.temp_root, exist_ok=True)
    device = args.device
    maptype = args.map_type
    OVERLAP_R = args.overlap_ratio
    ncol = args.num_column
    nrow = args.num_row
    r = args.nis_output_root
    # r = '/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4'
    for p in os.listdir(r):
        if '.' in p: continue                                   # skip files
        for b in os.listdir(f'{r}/{p}'):
            if '.' in b: continue                               # skip files
            if b.startswith('L'): continue                      # Add condition to skip unexpected brains
            fused_image_save_root = f'{r}/{p}'
            print(r, p, b)
            if not os.path.exists(f"{args.auto_stitch_root}/{p}/{b}/NIS_tranform/{b}_tform_refine.json"): continue
            # if os.path.exists(f"{fused_image_save_root}/fused:{maptype.replace(' ', '-')}_{b}.nii.gz"): continue
            if os.path.exists(f"{args.manual_stitch_root}/Manual_aligned_{b}"):
                stitch_type='Manual'
                manual_stitch_path = f"{args.manual_stitch_root}/Manual_aligned_{b}"
            else:
                stitch_type='Refine'
                manual_stitch_path = None

            nis_cpp_results = f'{r}/{p}/{b}'
            bm = Brainmap(cpp_result_root=nis_cpp_results, device=device, channel_layer=args.channel_layer,
                    maptype=maptype, # 'avg volume' or 'cell count'
                    ncol=ncol,
                    nrow=nrow,
                    org_overlap_r=OVERLAP_R,
                    btag_split=False)
            bm.arrange_brainmap_layer(args.stitch_root, stitch_type=stitch_type, manual_stitch_path=manual_stitch_path)  # Apply stitch transformation
            bm.run_fusion(save_root=args.temp_root) # Undouble NIS
            print('Saving to', fused_image_save_root)
            bm.save_fused_image(save_root=fused_image_save_root)# Save as .nii.gz

if __name__ == '__main__': main()