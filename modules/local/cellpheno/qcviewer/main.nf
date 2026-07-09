process CELLPHENO_QCVIEWER {
    tag "qc-viewer"
    label 'process_single'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://ghcr.io/chrisa142857/cellpheno-postproc:1.0.0' :
        'ghcr.io/chrisa142857/cellpheno-postproc:1.0.0' }"

    input:
    // Run-level trigger: the collected brain IDs (runs once, after the brain maps exist).
    val brains

    output:
    path "qcviewer/*", emit: bundle

    when:
    task.ext.when == null || task.ext.when

    script:
    // The cellpheno-viewer (https://github.com/Chrisa142857/cellpheno-viewer) is a static
    // niivue SPA + the nis_ondemand_viewer FastAPI backend, which serves zoom cubes
    // on-demand straight from the NIS results (no precompute). This step emits a
    // docker-compose + README to launch that backend against this run's outputs for
    // visual QC of stitching & segmentation from global to local.
    def img      = params.qcviewer_image
    def frontend = params.qcviewer_url
    def brainList = (brains instanceof List ? brains : [brains]).join(', ')
    """
    mkdir -p qcviewer

    cat > qcviewer/docker-compose.yml <<YML
    # Launch: CELLPHENO_RESULTS=/abs/path/to/<outdir> docker compose up -d
    services:
      nis-ondemand-viewer:
        image: ${img}
        environment:
          - NIS_ROOT=/results/nis
          - RAW_ROOT=/results/nis
          - STITCH_ROOT=/results/stitch
        volumes:
          - \${CELLPHENO_RESULTS:?set CELLPHENO_RESULTS to your --outdir}:/results:ro
        ports:
          - "8090:8090"
        command: uvicorn nis_ondemand_viewer.app:app --host 0.0.0.0 --port 8090
    YML

    cat > qcviewer/README.md <<MD
    # Visual QC viewer for this run

    Brains in this run: ${brainList}

    The [cellpheno-viewer](${frontend}) is a static niivue SPA backed by the
    \`nis_ondemand_viewer\` service, which serves brain maps + on-demand multi-scale
    zoom cubes **straight from the NIS results** (no precompute).

    1. Build/pull the backend image \`${img}\` (from the cellpheno-viewer repo,
       \`nis_ondemand_viewer/deploy/Dockerfile\`).
    2. From this directory:
       \`\`\`
       CELLPHENO_RESULTS=\$(realpath ${params.outdir}) docker compose up -d
       \`\`\`
    3. Open ${frontend} -> **Connect to server** -> \`http://localhost:8090\`.

    Note: the backend expects NIS results under \`<NIS_ROOT>/<pair>/<brain>/<tile>/\`.
    If your \`results/nis\` layout differs, adjust the env vars / mount above.
    MD

    cat <<-END_VERSIONS > qcviewer/versions.yml
    "${task.process}":
        cellpheno-qcviewer: 1.0.0
    END_VERSIONS
    """

    stub:
    """
    mkdir -p qcviewer
    echo "services: {}" > qcviewer/docker-compose.yml
    echo "# QC viewer bundle (stub)" > qcviewer/README.md
    echo 'v: 1.0.0' > qcviewer/versions.yml
    """
}
