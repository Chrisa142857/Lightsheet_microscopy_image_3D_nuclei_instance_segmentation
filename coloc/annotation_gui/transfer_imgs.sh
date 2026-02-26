mkdir -p /ssd2/userdata/ianraptor/Ian/Lightsheet/P14/male/pair3_test/Z800range

for zpos in $(seq 795 805); do
    find /ssd2/userdata/ianraptor/Ian/Lightsheet/P14/male/pair3/230810_L68P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_15-49-53 -type f -name "*_C01_*Z0${zpos}.ome.tif" -exec cp {} /ssd2/userdata/ianraptor/Ian/Lightsheet/P14/male/pair3_test/Z800range/ \;
done
