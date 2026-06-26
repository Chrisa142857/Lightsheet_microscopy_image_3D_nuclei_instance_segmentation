process NIS {
    tag "$meta.id"
    label 'process_high'
    label 'process_gpu'

    // NIS is a custom LibTorch + OpenCV + CUDA executable built from the source
    // repository (https://github.com/Chrisa142857/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation).
    // It has no Conda/Bioconda package, so it is distributed only as a dedicated
    // GPU container image. The conda directive is intentionally omitted because a
    // Conda environment cannot provide the prebuilt binary.
    container "${ workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container ?
        'oras://ghcr.io/chrisa142857/lightsheet-nis:1.0.0':
        'ghcr.io/chrisa142857/lightsheet-nis:1.0.0' }"

    input:
    tuple val(meta), path(tile_dir)
    path models

    output:
    tuple val(meta), path("*_NIScpp_results_*.zip"), emit: nis
    tuple val(meta), path("*_remap.zip")           , emit: remap, optional: true
    tuple val("${task.process}"), val('nis'), eval("cat /usr/local/share/nis/VERSION"), topic: versions, emit: versions_nis

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    // The GPU is selected by the executor (e.g. CUDA_VISIBLE_DEVICES); NIS defaults
    // to `cuda:0`. Override with `--device cuda:N` via `task.ext.args` if needed.
    """
    main \\
        --model_root ${models} \\
        --data_root ${tile_dir} \\
        --save_root . \\
        --brain_tag ${prefix} \\
        $args
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}_NIScpp_results_zmin0_seg_meta.zip
    touch ${prefix}_NIScpp_results_zmin0_binary_mask.zip
    touch ${prefix}_NIScpp_results_zmin0_instance_center.zip
    touch ${prefix}_NIScpp_results_zmin0_instance_coordinate.zip
    touch ${prefix}_NIScpp_results_zmin0_instance_label.zip
    touch ${prefix}_NIScpp_results_zmin0_instance_volume.zip
    touch ${prefix}_remap.zip
    """
}
