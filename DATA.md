# Data and large files

Redrawing the figures needs **nothing** beyond this repository: every number is cached under
`figures/paper/cache/`, so `make_figures.py` and the notebooks run with no downloads, no GPU, and
no whole-slide images.

The large files below are only needed to go **beyond** the cached figures, that is, to re-apply the
trained model, retrain it, or re-encode slides.

## One download from Baidu Netdisk

    链接 (link):   https://pan.baidu.com/s/1OR5YpYBC4rENwHBIcXnc7Q?pwd=akv6
    提取码 (code): akv6

Everything is in a single archive, `HNSC_data.tar` (about 16 GB). Its folders already mirror this
repository, so you just extract it **into the repository root** and every file lands where the code
expects it. No renaming, no moving:

    export HNSC_ROOT=/path/to/HNSCC-CellType-Expression-HE
    tar -xf HNSC_data.tar -C "$HNSC_ROOT"

That places:

    run/TCGA_HNSC_oro_v2b/results/     trained per-cell-type models + per-patient predictions (4.6 GB)
    run/TCGA_HNSC_oro_v2b/breast_*     deconvolved training labels (~0.5 GB)
    run/TCGA_HNSC/features.pkl         CTransPath tile features, TCGA  (3.4 GB)
    run/CPTAC_HNSC/features.pkl        CTransPath tile features, CPTAC (1.2 GB)
    run/HANCOCK/features.pkl           CTransPath tile features, HANCOCK (5.8 GB)

## Files obtained from their original sources (not on Baidu)

**Whole-slide images.** Public and too large to redistribute.
- TCGA-HNSC: NIH Genomic Data Commons, https://portal.gdc.cancer.gov/projects/TCGA-HNSC
- CPTAC-HNSCC: The Cancer Imaging Archive, https://www.cancerimagingarchive.net/collection/cptac-hnscc/ (also on the GDC)
- HANCOCK: https://hancock.research.fau.de

**CTransPath encoder weights.** Distributed by the original authors and not redistributed here.
Download `ctranspath.pth` from the TransPath repository (https://github.com/Xiyue-Wang/TransPath)
and point the pipeline at it with the `CTRANSPATH_WEIGHTS` variable (see `config.env.example`).

**Single-cell reference.** The oropharyngeal HNSCC atlas from CZ CELLxGENE, collection
3c34e6f1-6827-47dd-8e19-9edcd461893f, published as doi:10.1186/s12943-024-02191-9.
