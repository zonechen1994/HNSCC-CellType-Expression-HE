# Provenance of this directory

The model in this directory is **SLIDE-EX**, published by Wang et al., *npj Precision Oncology*,
2026, doi:10.1038/s41698-026-01419-9. It is not a contribution of the present study. We reuse it
to predict cell type-specific expression in head and neck cancer, and we thank its authors for
making the code available.

## From the original authors

    SLIDE_EX_code/prediction/                     the MLP model, training loops and data utilities
    SLIDE_EX_code/slide_processing/1main_processing.py
    SLIDE_EX_code/slide_processing/collect_features.py
    SLIDE_EX_code/slide_processing/collect_mask.py
    SLIDE_EX_code/slide_processing/func/          tiling, tissue detection and stain normalization

`func/utils_color_norm.py` implements Macenko stain normalization (Macenko et al., ISBI 2009).

## Added for this study

    SLIDE_EX_code/slide_processing/ctrans_model.py             CTransPath encoder wrapper
    SLIDE_EX_code/slide_processing/extract_features_parallel.py  multi-GPU sharded feature extraction

Both reproduce the original preprocessing exactly; the second one only shards the work across GPUs
so that three cohorts can be encoded in reasonable time. The CTransPath encoder itself is from
Wang et al., *Medical Image Analysis*, 2022 (https://github.com/Xiyue-Wang/TransPath); its weights
are not redistributed here, see `../../DATA.md`.

Everything outside this directory — the deconvolution pipeline that produces the training labels,
the three-cohort design, and all analyses and figures in the paper — is our own work. See the
Attribution section of the top-level `README.md`.
