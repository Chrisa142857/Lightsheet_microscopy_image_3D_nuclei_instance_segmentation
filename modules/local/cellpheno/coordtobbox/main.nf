process CELLPHENO_COORDTOBBOX {
    tag "$meta.id"
    label 'process_medium'
    label 'process_gpu'

    // Postprocessing (Python/torch) steps share one container that bundles this
    // repository's scripts; see containers/postproc/Dockerfile. No conda package.
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://ghcr.io/chrisa142857/cellpheno-postproc:1.0.0' :
        'ghcr.io/chrisa142857/cellpheno-postproc:1.0.0' }"

    input:
    tuple val(meta), path(nis_results)

    output:
    // One tar per tile holding the UltraII[col x row]/ dir (NIS zips + instance_bbox).
    // Passed as a tar because the literal dir name contains spaces and '[ ]', which
    // Nextflow path globs cannot represent.
    tuple val(meta), path("${meta.id}.tar"), emit: tile
    path "versions.yml"                    , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args    = task.ext.args ?: ''
    def device  = meta.device ?: 'cuda:0'
    def tx      = (meta.tile_x ?: 0) as Integer
    def ty      = (meta.tile_y ?: 0) as Integer
    def tile    = String.format('UltraII[%02d x %02d]', tx, ty)
    def VERSION = '1.0.0'
    """
    mkdir -p "${tile}"
    cp -L ${nis_results} "${tile}/"

    cellpheno_coordtobbox.py \\
        --tile-dir "${tile}" \\
        --out-dir "${tile}" \\
        --device ${device} \\
        ${args}

    tar -cf "${meta.id}.tar" "${tile}"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cellpheno-coordtobbox: ${VERSION}
    END_VERSIONS
    """

    stub:
    def tx      = (meta.tile_x ?: 0) as Integer
    def ty      = (meta.tile_y ?: 0) as Integer
    def tile    = String.format('UltraII[%02d x %02d]', tx, ty)
    def btag    = meta.brain ?: meta.id
    def VERSION = '1.0.0'
    """
    mkdir -p "${tile}"
    touch "${tile}/${btag}_NIScpp_results_zmin0_instance_bbox.zip"
    tar -cf "${meta.id}.tar" "${tile}"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cellpheno-coordtobbox: ${VERSION}
    END_VERSIONS
    """
}
