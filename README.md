# MechGate E. coli Permeation

Scores a molecule on five Escherichia coli barriers: OmpF porin permeation, multi-porin permeation, lipid-route permeation, efflux susceptibility and whole-cell activity (MIC of 8 ug/mL or lower), from 3D conformer and charge descriptors at pH 7.4. Models were trained on CO-ADD screens and ChEMBL; each score carries a percentile against 73,154 CO-ADD compounds, with four binary calls at the authors cut-offs. Validated retrospectively on marketed drugs. Output depends on the SMILES string written, and large molecules are slow.

This model was incorporated on 2026-10-29.Last packaged on 2026-10-30.

## Information
### Identifiers
- **Ersilia Identifier:** `eos8bpb`
- **Slug:** `ecoli-permeation-efflux`

### Domain
- **Task:** `Annotation`
- **Subtask:** `Activity prediction`
- **Biomedical Area:** `Antimicrobial resistance`
- **Target Organism:** `Escherichia coli`
- **Tags:** `Antimicrobial activity`, `Gram-negative bacteria`, `Permeability`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `14`
- **Output Consistency:** `Fixed`
- **Interpretation:** Five E. coli measures as scores and CO-ADD percentiles, plus four calls where 1 means high or selected.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| ompf_score | float | high | Fixed-formula OmpF porin permeation score from molecular size and net charge and primary amine count where higher means more favourable passage |
| ompf_percentile | float | high | Percentile of the OmpF score among 73154 CO-ADD reference molecules where higher means more favourable passage |
| ompf_cut | integer | high | Call of 1 when the OmpF percentile is at least 0.95 and 0 otherwise |
| multiporin_score | float | high | Random forest multi-porin permeation score where higher means more favourable passage |
| multiporin_percentile | float | high | Percentile of the multi-porin score among 73154 CO-ADD reference molecules where higher means more favourable passage |
| multiporin_cut | integer | high | Call of 1 when the multi-porin percentile is at least 0.98 and 0 otherwise |
| lipid_score | float | high | Random forest probability from the CO-ADD lpxC case-control that the outer membrane holds the molecule back where higher means more held back |
| lipid_percentile | float | high | Percentile of the reversed lipid-route score among 73154 CO-ADD reference molecules where higher means the outer membrane holds the molecule back less |
| lipid_cut | integer | high | Call of 1 when the lipid-route percentile is at least 0.99 and 0 otherwise |
| efflux_score | float | high | Logistic regression probability from the CO-ADD tolC case-control that the molecule is pumped out where higher means more efflux |

_10 of 14 columns are shown_
### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos8bpb](https://hub.docker.com/r/ersiliaos/eos8bpb)
- **Docker Architecture:** `AMD64`, `ARM64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos8bpb.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos8bpb.zip)

### Resource Consumption
- **Model Size (Mb):** `5`
- **Environment Size (Mb):** `1587`
- **Image Size (Mb):** `1710.52`

**Computational Performance (seconds):**
- 10 inputs: `24.93`
- 100 inputs: `358.61`
- 10000 inputs: `-1`

### References
- **Source Code**: [https://github.com/qinlang0008-blip/mechgate](https://github.com/qinlang0008-blip/mechgate)
- **Publication**: [https://doi.org/10.48550/arXiv.2610.33809](https://doi.org/10.48550/arXiv.2610.33809)
- **Publication Type:** `Preprint`
- **Publication Year:** `2026`
- **Ersilia Contributor:** [TiagoJanela](https://github.com/TiagoJanela)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [GPL-2.0-only](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos8bpb
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos8bpb
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
