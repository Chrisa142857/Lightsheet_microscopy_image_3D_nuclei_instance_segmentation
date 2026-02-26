import os
import psycopg2
import numpy as np
import h5py
from datetime import datetime

DB_CONFIG = {
    "dbname": "cellpheno_db",
    "user": "ziquanw",
    # "password": "your_password",
    "host": "localhost",
    "port": "5432"
}
pair_tag = 'pair21'
brain_tag = '220923_L91D814P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_16-24-18'
SCAN_DIR1 = f"/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{pair_tag}/{brain_tag}"
SCAN_DIR2 = f"/cajal/Felix/Lightsheet/P4/{pair_tag}/{brain_tag}"
if not os.path.exists(SCAN_DIR2):
    SCAN_DIR2 = f"/lichtman/Felix/Lightsheet/P4/{pair_tag}/{brain_tag}"
BRAIN_NAME = brain_tag.split('_')[1]
# Connect to PostgreSQL
conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()
# Create table if not exists
cursor.execute(f'''
CREATE TABLE {BRAIN_NAME} (
    id SERIAL PRIMARY KEY,
    path TEXT UNIQUE,
    bbox geometry(POLYGONZ, 4326), -- 3D bounding box
    shape TEXT,  -- Patch dimensions (e.g., "(64, 64, 32)")
    file_type TEXT,
    size BIGINT,
    created_at TIMESTAMP,
    modified_at TIMESTAMP
);
CREATE INDEX {BRAIN_NAME}_gix ON {BRAIN_NAME} USING GIST (bbox);
''')

def extract_metadata(filepath):
    """Extract metadata without loading full data into memory."""
    file_type = filepath.split('.')[-1]
    shape = None
    coords = (0, 0, 0, 0, 0, 0)  # Default (x_start, x_end, y_start, y_end, z_start, z_end)

    try:
        if file_type == "npy":
            array = np.load(filepath, mmap_mode='r')
            shape = array.shape
        elif file_type == "h5":
            with h5py.File(filepath, 'r') as f:
                dataset_name = list(f.keys())[0]
                array = f[dataset_name]
                shape = array.shape

        filename = os.path.basename(filepath)
        coords = tuple(map(int, filename.replace(file_type, "").split("_")))

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return coords, shape, file_type

def insert_patch(filepath):
    """Insert patch metadata into PostgreSQL."""
    try:
        stat = os.stat(filepath)
        coords, shape, file_type = extract_metadata(filepath)
        x1, x2, y1, y2, z1, z2 = coords

        # Define a 3D bounding box (POLYGONZ)
        bbox = f"POLYGONZ(({x1} {y1} {z1}, {x2} {y1} {z1}, {x2} {y2} {z1}, {x1} {y2} {z1}, {x1} {y1} {z1}), \
                        ({x1} {y1} {z2}, {x2} {y1} {z2}, {x2} {y2} {z2}, {x1} {y2} {z2}, {x1} {y1} {z2}))"

        query = f'''
        INSERT INTO {BRAIN_NAME} (path, bbox, shape, file_type, size, created_at, modified_at)
        VALUES (%s, ST_GeomFromText(%s, 4326), %s, %s, %s, %s, %s)
        ON CONFLICT (path) DO NOTHING;
        '''
        cursor.execute(query, (filepath, bbox, str(shape), file_type, stat.st_size,
                               datetime.fromtimestamp(stat.st_ctime), datetime.fromtimestamp(stat.st_mtime)))
    except Exception as e:
        print(f"Error inserting {filepath}: {e}")

def scan_directory(directory):
    """Traverse filesystem and store metadata in PostgreSQL."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip') and 'instance' in file:  
                insert_patch(os.path.join(root, file))
            elif file.endswith('.ome.tif'):
                insert_patch(os.path.join(root, file))

scan_directory(SCAN_DIR1)

conn.commit()
conn.close()
print("PostgreSQL database populated successfully!")
