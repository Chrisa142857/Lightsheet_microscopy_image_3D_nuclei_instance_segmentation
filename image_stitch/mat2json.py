
# r = '/lichtman/Ian/Lightsheet/P14/stitched/female/L106P3/variables'
r = '/lichtman/Ian/Lightsheet/P14/stitched/female/L102P2/variables'

import subprocess

def f(z_tform_fn='adjusted_z.mat', tgt_key='z_adj'):
    z_tform_path = f'{r}/{z_tform_fn}'
    save_fn = z_tform_fn.replace('.mat', '.json')
    save_path = f'numorph_param/{r.split("/")[-2]}_{save_fn}'
    # Path to the MATLAB executable (may vary based on your system and installation)
    matlab_executable = "matlab"  # Use 'matlab' or the full path to MATLAB executable if needed

    # Command to run the MATLAB script without GUI (in batch mode)
    command = [
        matlab_executable,
        "-nodisplay",   # Run without GUI
        "-nosplash",    # Disable the splash screen
        "-r",           # Run the following MATLAB command
        f'load("{z_tform_path}"); z_json = jsonencode({tgt_key}); fid = fopen("{save_path}","w"); fprintf(fid, "%s",z_json); fclose(fid); exit;'  # Command to run your script and exit MATLAB after
    ]

    # Execute the command
    print(command)
    subprocess.run(command)

f('adjusted_z.mat', 'z_adj')