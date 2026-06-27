process CELLPHENO_STITCH {
    tag "$meta.id"
    label 'process_high'
    label 'process_gpu'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://ghcr.io/chrisa142857/cellpheno-postproc:1.0.0' :
        'ghcr.io/chrisa142857/cellpheno-postproc:1.0.0' }"

    input:
    // meta is the BRAIN meta (id == brain tag). tile_tars = all per-tile tars of the
    // brain (from CELLPHENO_COORDTOBBOX, grouped). image_dir = raw light-sheet images.
    tuple val(meta), path(tile_tars), path(image_dir)

    output:
    tuple val(meta), path("NIS_tranform/${meta.id}_tform_refine.json"), emit: tform
    path "versions.yml"                                               , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args    = task.ext.args ?: ''
    def ptag    = meta.pair ?: 'pair'
    def btag    = meta.id
    def overlap = meta.overlap_r ?: params.overlap_ratio
    def VERSION = '1.0.0'
    """
    # Reassemble the brain's tiles (UltraII[col x row]/) from the per-tile tars.
    for t in *.tar; do tar -xf "\$t"; done

    cellpheno_stitch.py \\
        --ptag ${ptag} \\
        --btag ${btag} \\
        --ls-image-root ${image_dir} \\
        --nis-result-path . \\
        --save-path . \\
        --overlap-r ${overlap} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cellpheno-stitch: ${VERSION}
    END_VERSIONS
    """

    stub:
    def VERSION = '1.0.0'
    """
    mkdir -p NIS_tranform
    echo '{}' > "NIS_tranform/${meta.id}_tform_refine.json"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cellpheno-stitch: ${VERSION}
    END_VERSIONS
    """
}
