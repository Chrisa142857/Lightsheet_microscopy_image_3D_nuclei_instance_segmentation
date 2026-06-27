process NIS_SEGMENT {
    tag "$meta.id"
    label 'process_high'
    label 'process_gpu'

    // NIS is a custom LibTorch + OpenCV + CUDA executable built from this
    // repository (see containers/nis/Dockerfile). It has no bioconda package,
    // so it is distributed as a dedicated container image rather than via conda
    // (no `conda` directive: a conda env cannot provide the prebuilt binary).
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://ghcr.io/chrisa142857/cellpheno-nis:1.0.0' :
        'ghcr.io/chrisa142857/cellpheno-nis:1.0.0' }"

    input:
    tuple val(meta), path(tile_dir)
    path models

    output:
    tuple val(meta), path("*_NIScpp_results_*.zip"), emit: nis
    tuple val(meta), path("*_remap.zip")           , emit: remap, optional: true
    path "versions.yml"                            , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    // `meta.device` lets a pipeline pin the GPU (e.g. 'cuda:0'); falls back to cuda:0.
    def device = meta.device ?: 'cuda:0'
    // Single source of truth for the reported NIS version. NIS has no parseable
    // `--version` output, so report the release/container tag. Keep in sync with
    // the container tag above and `manifest.version` in nextflow.config.
    def VERSION = '1.0.0'
    """
    main \\
        --device ${device} \\
        --model_root ${models} \\
        --data_root ${tile_dir} \\
        --save_root . \\
        --brain_tag ${prefix} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nis: ${VERSION}
    END_VERSIONS
    """

    stub:
    def prefix  = task.ext.prefix ?: "${meta.id}"
    def VERSION = '1.0.0'
    """
    touch ${prefix}_NIScpp_results_zmin0_seg_meta.zip
    touch ${prefix}_NIScpp_results_zmin0_binary_mask.zip
    touch ${prefix}_NIScpp_results_zmin0_instance_center.zip
    touch ${prefix}_NIScpp_results_zmin0_instance_coordinate.zip
    touch ${prefix}_NIScpp_results_zmin0_instance_label.zip
    touch ${prefix}_NIScpp_results_zmin0_instance_volume.zip
    touch ${prefix}_remap.zip

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nis: ${VERSION}
    END_VERSIONS
    """
}
