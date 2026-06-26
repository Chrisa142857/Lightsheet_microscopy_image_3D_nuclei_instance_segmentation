/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT MODULES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { NIS_SEGMENT } from '../modules/local/nis/main'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow NIS {

    main:
    ch_versions = Channel.empty()

    if (!params.input)  { error "Please provide a samplesheet with --input (columns: sample,tile_dir[,device])" }
    if (!params.models) { error "Please provide the TorchScript model directory with --models" }

    ch_models = file(params.models, checkIfExists: true)

    // Samplesheet: one row per tile -> [ meta, tile_dir ]
    ch_tiles = Channel
        .fromPath(params.input, checkIfExists: true)
        .splitCsv(header: true)
        .map { row ->
            if (!row.sample || !row.tile_dir) {
                error "Samplesheet must contain 'sample' and 'tile_dir' columns"
            }
            def meta = [ id: row.sample ]
            if (row.device) { meta.device = row.device }
            [ meta, file(row.tile_dir, checkIfExists: true) ]
        }

    NIS_SEGMENT ( ch_tiles, ch_models )
    ch_versions = ch_versions.mix(NIS_SEGMENT.out.versions.first())

    emit:
    nis      = NIS_SEGMENT.out.nis
    remap    = NIS_SEGMENT.out.remap
    versions = ch_versions
}
