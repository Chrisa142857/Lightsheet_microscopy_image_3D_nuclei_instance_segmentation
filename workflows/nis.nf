/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT MODULES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { NIS_SEGMENT           } from '../modules/local/nis/main'
include { CELLPHENO_COORDTOBBOX } from '../modules/local/cellpheno/coordtobbox/main'
include { CELLPHENO_STITCH      } from '../modules/local/cellpheno/stitch/main'
include { CELLPHENO_BRAINMAP    } from '../modules/local/cellpheno/brainmap/main'
include { CELLPHENO_COLOC       } from '../modules/local/cellpheno/coloc/main'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW

    One `nextflow run` chains the whole whole-brain pipeline that used to be a set
    of manual command lines (cpp/README.md + repo README):

      NIS (per tile) -> coord_to_bbox (per tile)
                     -> [group tiles by brain]
                     -> stitch -> brain map -> (optional) coloc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow NIS {

    main:
    ch_versions = Channel.empty()

    if (!params.input)  { error "Please provide a samplesheet with --input (columns: brain,pair,tile_x,tile_y,tile_dir,image_dir[,device])" }
    if (!params.models) { error "Please provide the TorchScript model directory with --models" }

    ch_models = file(params.models, checkIfExists: true)

    // Samplesheet: one row per TILE of a brain.
    ch_tiles = Channel
        .fromPath(params.input, checkIfExists: true)
        .splitCsv(header: true)
        .map { row ->
            if (!row.brain || !row.tile_dir) {
                error "Samplesheet must contain at least 'brain' and 'tile_dir' columns"
            }
            def tx = (row.tile_x ?: 0) as Integer
            def ty = (row.tile_y ?: 0) as Integer
            def meta = [
                id     : "${row.brain}_t${tx}x${ty}".toString(),
                brain  : row.brain,
                pair   : row.pair ?: 'pair',
                tile_x : tx,
                tile_y : ty,
            ]
            if (row.image_dir) { meta.image_dir = row.image_dir }
            if (row.device)    { meta.device    = row.device }
            [ meta, file(row.tile_dir, checkIfExists: true) ]
        }

    //
    // 1) NIS segmentation per tile. The output prefix is the *brain* tag (set via
    //    ext.prefix in conf/modules.config) so each tile yields {btag}_NIScpp_* —
    //    the naming the stitch/brain-map code expects inside UltraII[col x row]/.
    //
    NIS_SEGMENT ( ch_tiles, ch_models )
    ch_versions = ch_versions.mix(NIS_SEGMENT.out.versions.first())

    //
    // 2) coord -> instance bounding boxes, per tile. Emits one tar per tile holding
    //    the UltraII[col x row]/ directory (NIS zips + instance_bbox).
    //
    CELLPHENO_COORDTOBBOX ( NIS_SEGMENT.out.nis )
    ch_versions = ch_versions.mix(CELLPHENO_COORDTOBBOX.out.versions.first())

    //
    // Group the per-tile tars by brain -> [ brain_meta, [ tile tars ] ].
    //
    ch_by_brain = CELLPHENO_COORDTOBBOX.out.tile
        .map { meta, tar -> [ meta.brain, meta, tar ] }
        .groupTuple()
        .map { brain, metas, tars ->
            def first = metas[0]
            def ncol  = (metas.collect { it.tile_x }.max() ?: 0) + 1
            def nrow  = (metas.collect { it.tile_y }.max() ?: 0) + 1
            def bmeta = [
                id        : brain.toString(),
                pair      : first.pair,
                ncol      : ncol,
                nrow      : nrow,
                image_dir : first.image_dir,
                device    : first.device,
            ]
            [ bmeta, tars ]
        }

    //
    // 3) Stitch transform per brain (needs the raw light-sheet image dir).
    //
    ch_stitch_in = ch_by_brain.map { bmeta, tars ->
        if (!bmeta.image_dir) {
            error "Brain '${bmeta.id}' needs an 'image_dir' column (raw light-sheet images) for stitching"
        }
        [ bmeta, tars, file(bmeta.image_dir, checkIfExists: true) ]
    }
    CELLPHENO_STITCH ( ch_stitch_in )
    ch_versions = ch_versions.mix(CELLPHENO_STITCH.out.versions.first())

    //
    // 4) Whole-brain map per brain: join grouped tars with the stitch transform.
    //
    ch_brainmap_in = ch_by_brain
        .map { bmeta, tars -> [ bmeta.id, bmeta, tars ] }
        .join( CELLPHENO_STITCH.out.tform.map { bmeta, tform -> [ bmeta.id, tform ] } )
        .map { id, bmeta, tars, tform -> [ bmeta, tars, tform ] }
    CELLPHENO_BRAINMAP ( ch_brainmap_in )
    ch_versions = ch_versions.mix(CELLPHENO_BRAINMAP.out.versions.first())

    //
    // 5) (Optional) NIS-guided multi-channel co-localization per brain.
    //
    ch_coloc = Channel.empty()
    if (params.run_coloc) {
        ch_weights = params.coloc_weights ? file(params.coloc_weights, checkIfExists: true) : []
        CELLPHENO_COLOC ( ch_by_brain, ch_weights )
        ch_coloc    = CELLPHENO_COLOC.out.coloc
        ch_versions = ch_versions.mix(CELLPHENO_COLOC.out.versions.first())
    }

    emit:
    nis      = NIS_SEGMENT.out.nis
    bbox     = CELLPHENO_COORDTOBBOX.out.tile
    tform    = CELLPHENO_STITCH.out.tform
    brainmap = CELLPHENO_BRAINMAP.out.brainmap
    coloc    = ch_coloc
    versions = ch_versions
}
