from tqdm import tqdm
import win32com.client
import pythoncom
import os

cwd = os.getcwd()
def win_shortcut(shortcut_path, target):
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(shortcut_path)
    shortcut.Targetpath = f'{cwd}\\{target}'
    shortcut.WindowStyle = 7 # 7 - Minimized, 3 - Maximized, 1 - Normal
    shortcut.save()

r = 'Z0750-Z1000'
z_portion = 50

for d in tqdm(sorted(os.listdir(r))):
    if 'Ultra' not in d: continue
    for fn in sorted(os.listdir(f'{r}/{d}')):
        if not fn.endswith('.ome.tif'): continue
        z = int(fn.split('_')[1])
        if z % z_portion == 0:
            save_d = f'Z{z:04d}-Z{z+z_portion:04d}/{d}'
            os.makedirs(save_d, exist_ok=True)
        win_shortcut(f'{save_d}/{fn}.lnk', f'{r}/{d}/{fn}')
