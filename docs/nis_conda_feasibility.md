# Can NIS be packaged for Conda/Bioconda? — Feasibility assessment

Context: nf-core/modules requires every tool to be installable from a Conda
package (so a BioContainer/Seqera image can be auto-built). NIS currently fails
that check because it is a custom C++/LibTorch/CUDA GPU binary with no Conda
package. This note assesses whether that gap can realistically be closed.

> **Resolution (no maintainers needed).** The question this note framed as "ask the
> maintainers whether a container-only GPU module is acceptable" is already answered
> by precedent: the merged **`nf-core/parabricks/*`** modules are GPU-only,
> container-only (`nvcr.io`), and ship **no conda package**. Our `modules/nf-core/nis`
> now follows that exact pattern and lints **41/41** — the same posture as
> `parabricks/fq2bam`. So Bioconda packaging is **not required** at all; the analysis
> below remains as the record of *why* packaging wouldn't have helped anyway.

## TL;DR

- **Building the NIS binary as a Conda package is technically feasible** — every
  build dependency exists on conda-forge, including CUDA-enabled LibTorch.
- **But it does NOT unblock nf-core**, because the auto-generated
  BioContainer/Seqera image is assembled on CPU-only infrastructure and would not
  be GPU-functional. A custom GPU container is still required to *run* NIS.
- **Recommendation:** keep the dedicated GPU container for execution and ask the
  nf-core maintainers to accept a container-only GPU module. A conda-forge package
  is at best a "nice to have" for discoverability — it is not a true unblock.

## What works in favour (the binary *can* be built)

All build blocks are available on conda-forge, an allowed channel for Bioconda
recipes:

| Dependency | conda-forge availability |
| --- | --- |
| LibTorch (CUDA) | ✅ `libtorch` with `cuda128` / `cuda129` / `cuda130` variants |
| OpenCV | ✅ `opencv` |
| Build tools | ✅ `cmake`, `cuda-nvcc`, `cuda-toolkit`, C/C++ compilers |

Other favourable points:

- Compiling/linking NIS against CUDA LibTorch is a **link-time** operation — it
  does **not** require a GPU to be present, so it can run on Bioconda's CPU CI.
- The TorchScript `.pt` models are **runtime data** (analogous to reference
  databases): they are fetched separately and would not be part of the package.

## What works against it (why Bioconda is a poor fit)

1. **CPU-only build/test/container infrastructure.** Bioconda's CI agents have no
   GPU. The package could *build*, but the mandatory recipe smoke-test cannot
   exercise the tool — and NIS has no working `--version` to even do a trivial
   check.
2. **The auto-container would not be GPU-capable — this is the crux.** nf-core
   pulls the BioContainer/Seqera image that is auto-built from the Conda package
   on CPU infrastructure, without the CUDA runtime/driver wiring. So even *with* a
   Conda package, nf-core's container path still would not run NIS on a GPU.
   Packaging therefore does not remove the blocker it was meant to remove.
3. **GPU packages are rare and awkward on Bioconda.** Bioconda targets CPU
   bioinformatics CLIs; GPU/CUDA requests (e.g. bioconda-recipes#16358, "GPU
   Racon") hit friction. Acceptance is not guaranteed.
4. **Device/model coupling.** NIS only supports `cuda:0`/`cuda:1` and needs
   device-specific TorchScript models (e.g. `grad_2Dto3D_cuda:0.pt`). This is a
   packaging smell, though it does not block building the binary itself.
5. **Release + size.** A recipe builds from a tagged source tarball (this repo has
   no releases yet), and CUDA-LibTorch builds are large — they may hit Bioconda CI
   time/resource limits.

## conda-forge vs Bioconda

conda-forge has first-class CUDA support and already ships CUDA LibTorch, so a
generic GPU C++ tool is a more natural fit there than on Bioconda. nf-core
`environment.yml` files already use the conda-forge channel. However, the
"auto-container is not GPU-ready" problem (point 2) applies equally to a
conda-forge package — so it still does not give nf-core a working GPU image.

## Conclusion

There is no path where Conda packaging *alone* produces a working nf-core GPU
module: the auto-build container model is CPU-oriented by design. The realistic
options to put to the maintainers are therefore about the **container policy**,
not about packaging — which is what the issue #11311 comment proposes.

Sources: conda-forge `libtorch` CUDA variants (anaconda.org/conda-forge/libtorch);
Bioconda contributor guidelines (bioconda.github.io/contributor/guidelines.html);
bioconda-recipes#16358 (GPU Racon); nf-core Conda recommendations
(nf-co.re/docs/guidelines/pipelines/recommendations/bioconda).
