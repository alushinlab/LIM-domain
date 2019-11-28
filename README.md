# LIM domain analyses

This repository contains tools to quantify the cellular partition of proteins
and analyze protein binding properties in *in vitro* reconstitution assays.

`actin_enrich.py` quantifies the F-actin and nuclear enrichments of a
protein on a whole-cell basis, and is suitable to analyze epifluorescence or
confocal images.

`sf_enrich.py` quantifies the stress-fiber enrichment of a protein and
is suitable to analyze confocal images. It takes as input an ROI mask
containing the stress fiber and its local cytosol, as well as a maximum
intensity projection of a sub-stack containing the stress fiber.

`nuc_enrich_fixed.py` quantifies the nuclear enrichment of a protein
in fixed cells using confocal z-stack images. The nuclear enrichment (with
respect to either the whole-cell cytosol or the local cytocol proximal to the
nucleus) is computed based on the slice with the maximum nuclear area.

`nuc_enrich_live.py` quantifies the nuclear enrichment of a protein in
live cells over time using time-lapse confocal images. This script can be used
to analyze the effect of a drug treatment and wash-out.

`patch.py` detects proteins that bind to F-actin in in vitro
reconstitution assay monitored by TIRF microscopy and tracks the binding
patches over time. It computes a variety of properties related to the
binding patches, such as lifetime, intensity, and area. It is suitable to
analyze images with high background and non-uniform illumination.

To install the dependencies, run:
```bash
pip install -r requirements.txt
```
