process CELLPHENO_MORPHOMETRY {
    tag "$meta.id"
    // CPU-only: SimpleITK LabelShapeStatisticsImageFilter per instance (no CUDA).
    label 'process_medium'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://ghcr.io/chrisa142857/cellpheno-postproc:1.0.0' :
        'ghcr.io/chrisa142857/cellpheno-postproc:1.0.0' }"

    input:
    tuple val(meta), path(nis_results)

    output:
    tuple val(meta), path("*instance_pa*.zip"), emit: morphometry
    path "versions.yml"                        , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args    = task.ext.args ?: ''
    def vmin    = params.morph_vol_min
    def vmax    = params.morph_vol_max
    def VERSION = '1.0.0'
    """
    cellpheno_morphometry.py \\
        --tile-dir . \\
        --out-dir . \\
        --vol-min ${vmin} \\
        --vol-max ${vmax} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cellpheno-morphometry: ${VERSION}
    END_VERSIONS
    """

    stub:
    def btag    = meta.brain ?: meta.id
    def VERSION = '1.0.0'
    """
    touch ${btag}_NIScpp_results_zmin0_instance_pa.zip

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cellpheno-morphometry: ${VERSION}
    END_VERSIONS
    """
}
