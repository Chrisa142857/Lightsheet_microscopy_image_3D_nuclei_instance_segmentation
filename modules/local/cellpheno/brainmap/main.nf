process CELLPHENO_BRAINMAP {
    tag "$meta.id"
    label 'process_high'
    label 'process_gpu'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://ghcr.io/chrisa142857/cellpheno-postproc:1.0.0' :
        'ghcr.io/chrisa142857/cellpheno-postproc:1.0.0' }"

    input:
    // meta is the BRAIN meta. tile_tars = per-tile tars (grouped); tform = stitch json.
    tuple val(meta), path(tile_tars), path(tform)

    output:
    tuple val(meta), path("*.nii.gz"), emit: brainmap
    path "versions.yml"              , emit: versions

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
    def maptype = params.map_type
    def VERSION = '1.0.0'
    """
    # Rebuild the {pair}/{brain} layout the Brainmap class expects (it parses
    # ptag/btag from the last two path components) and place the stitch transform.
    mkdir -p "nis/${ptag}/${btag}" "stitch/${ptag}/${btag}/NIS_tranform"
    for t in *.tar; do tar -xf "\$t" -C "nis/${ptag}/${btag}"; done
    cp -L ${tform} "stitch/${ptag}/${btag}/NIS_tranform/"

    cellpheno_brainmap.py \\
        --nis-result-path "nis/${ptag}/${btag}" \\
        --stitch-root stitch \\
        --map-type "${maptype}" \\
        --ncol ${ncol} \\
        --nrow ${nrow} \\
        --overlap-r ${overlap} \\
        --device ${device} \\
        --save-root . \\
        --temp-root ./tmp \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cellpheno-brainmap: ${VERSION}
    END_VERSIONS
    """

    stub:
    def maptype = (params.map_type ?: 'cell count').replace(' ', '-')
    def VERSION = '1.0.0'
    """
    touch "fused_${maptype}_${meta.id}.nii.gz"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cellpheno-brainmap: ${VERSION}
    END_VERSIONS
    """
}
