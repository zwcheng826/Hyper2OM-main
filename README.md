# Hyper2OM Information Fusion

An RNA 2'-O-methylation site prediction model that integrates RiNALMo features, local sequence features, a shared static hypergraph, and a Transformer.

## Files

- `train.py`: Train the model with cross-validation and save checkpoints.
- `test.py`: Load fold checkpoints and generate ensemble predictions.
- `extract_rinalmo_features.py`: Extract sequence features with RiNALMo.

## Directory Structure

```text
Hyper2OM_InformationFusion/
├── data/       # CSV datasets
├── features/   # RiNALMo features
├── rinalmo/    # RiNALMo source code
├── weights/    # Pretrained weights
├── results/    # Checkpoints and predictions
├── train.py
├── test.py
└── extract_rinalmo_features.py
```

Default dataset filenames:

```text
{A,U,C,G}_H2Opred_{Tr,Te}.csv
```

Each CSV file must contain the sequence and label fields required by the scripts. RNA sequences should use uppercase `A/U/C/G`.

## Environment

```bash
pip install torch numpy pandas scikit-learn scipy tqdm
```

Place the RiNALMo source code in `rinalmo/` and the pretrained checkpoint at:

```text
weights/rinalmo_giga_pretrained.pt
```

## Usage

Extract RiNALMo features:

```bash
python extract_rinalmo_features.py
```

Train the model:

```bash
python train.py --data_root ./data --output_root ./results
```

Test the model:

```bash
python test.py --data_root ./data --output_root ./results
```

To overwrite existing training results:

```bash
python train.py --force
```

Do not commit datasets, pretrained weights, cached features, or experiment results to GitHub unless their licenses and file sizes permit redistribution. Add the appropriate license and third-party citations before publishing the repository.
