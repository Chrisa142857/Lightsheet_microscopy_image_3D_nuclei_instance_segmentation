process CELLPHENO_STITCHREFINE {
    tag "$meta.id"
    label 'process_high'
    label 'process_gpu'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://ghcr.io/chrisa142857/cellpheno-postproc:1.0.0' :
        'ghcr.io/chrisa142857/cellpheno-postproc:1.0.0' }"

    input:
    // Brain meta + per-tile tars (grouped). Optional README step 3.5: a finer,
    // point-registration stitch transform that brainmap uses in preference if present.
    tuple val(meta), path(tile_tars)

    output:
    tuple val(meta), path("NIS_tranform/${meta.id}_tform_refine_ptreg.json"), emit: tform
    path "versions.yml"                                                     , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args    = task.ext.args ?: ''
    def device  = meta.device ?: 'cuda:0'
    def ptag    = meta.pair ?: 'pair'
    def btag    = meta.id
    def ncol    = meta.ncol ?: params.num_column
    def nrow    = meta.nrow ?: params.num_row
    def overlap = meta.overlap_r ?: params.overlap_ratio
    def zrange  = params.ptreg_zrange
    def VERSION = '1.0.0'
    """
    for t in *.tar; do tar -xf "\$t"; done

    cellpheno_stitchrefine.py \\
        --ptag ${ptag} \\
        --btag ${btag} \\
        --nis-result-path . \\
        --save-path . \\
        --ncol ${ncol} \\
        --nrow ${nrow} \\
        --overlap-r ${overlap} \\
        --zrange ${zrange} \\
        --device ${device} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cellpheno-stitchrefine: ${VERSION}
    END_VERSIONS
    """

    stub:
    def VERSION = '1.0.0'
    """
    mkdir -p NIS_tranform
    echo '{}' > "NIS_tranform/${meta.id}_tform_refine_ptreg.json"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cellpheno-stitchrefine: ${VERSION}
    END_VERSIONS
    """
}
