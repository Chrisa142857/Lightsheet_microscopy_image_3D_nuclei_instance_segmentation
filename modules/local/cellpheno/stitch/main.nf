process CELLPHENO_STITCH {
    tag "$meta.id"
    // CPU-only: get_stitch_tform uses skimage phase-correlation + CPU torch interpolation
    // (no CUDA). Memory-heavy — it preloads ncol*nrow raw 3D image stacks.
    label 'process_high'

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
    def ncol    = meta.ncol ?: params.num_column
    def nrow    = meta.nrow ?: params.num_row
    def pattern = params.image_tile_pattern ?: 'UltraII[%02d x %02d]'
    def slicep  = params.slice_filename_pattern ? "--slice-pattern '${params.slice_filename_pattern}'" : ''
    def VERSION = '1.0.0'
    """
    # Reassemble the brain's tiles (UltraII[col x row]/) from the per-tile tars.
    for t in *.tar; do tar -xf "\$t"; done

    # Map the raw-image tile folders (customizable --image_tile_pattern) to the
    # UltraII[col x row] layout get_stitch_tform expects, leaving the research code
    # untouched. The pattern receives column then row as two integer printf args.
    mkdir -p ls_image
    for i in \$(seq 0 \$(( ${ncol} - 1 ))); do
      for j in \$(seq 0 \$(( ${nrow} - 1 ))); do
        src=\$(printf '${pattern}' "\$i" "\$j")
        dst=\$(printf 'UltraII[%02d x %02d]' "\$i" "\$j")
        if [ -e "${image_dir}/\$src" ]; then
          ln -s "../${image_dir}/\$src" "ls_image/\$dst"
        else
          echo "WARN: raw-image tile folder not found: ${image_dir}/\$src" >&2
        fi
      done
    done

    cellpheno_stitch.py \\
        --ptag ${ptag} \\
        --btag ${btag} \\
        --ls-image-root ls_image \\
        --nis-result-path . \\
        --save-path . \\
        --overlap-r ${overlap} \\
        ${slicep} \\
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
