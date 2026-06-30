process CELLPHENO_COLOC {
    tag "$meta.id"
    label 'process_high'
    label 'process_gpu'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://ghcr.io/chrisa142857/cellpheno-postproc:1.0.0' :
        'ghcr.io/chrisa142857/cellpheno-postproc:1.0.0' }"

    input:
    // meta is the BRAIN meta. tile_tars = per-tile tars (grouped). weights = optional
    // classifier checkpoint ([] when not provided -> auto-discovery in coloc/model_weights).
    tuple val(meta), path(tile_tars)
    path weights

    output:
    tuple val(meta), path("*_results_*.csv"), emit: coloc, optional: true
    path "versions.yml"                     , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args        = task.ext.args ?: ''
    def device      = meta.device ?: 'cuda:0'
    def ptag        = meta.pair ?: 'pair'
    def btag        = meta.id
    def gtag        = params.coloc_gtag
    def weights_arg = weights ? "--weights-path ${weights}" : ''
    def VERSION     = '1.0.0'
    """
    for t in *.tar; do tar -xf "\$t"; done

    cellpheno_coloc.py \\
        --ptag ${ptag} \\
        --btag ${btag} \\
        --gtag ${gtag} \\
        ${weights_arg} \\
        --device ${device} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cellpheno-coloc: ${VERSION}
    END_VERSIONS
    """

    stub:
    def VERSION = '1.0.0'
    """
    touch "${meta.pair ?: 'pair'}_${meta.id}_resnet50_results_Z0200-0400.csv"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cellpheno-coloc: ${VERSION}
    END_VERSIONS
    """
}
