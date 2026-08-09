# <span style="font-size:larger;">MACE</span>

[![GitHub release](https://img.shields.io/github/release/ACEsuit/mace.svg)](https://GitHub.com/ACEsuit/mace/releases/)
[![Paper](https://img.shields.io/badge/Paper-NeurIPs2022-blue)](https://openreview.net/forum?id=YPpSngE-ZU)
[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/mit)
[![GitHub issues](https://img.shields.io/github/issues/ACEsuit/mace.svg)](https://GitHub.com/ACEsuit/mace/issues/)
[![Documentation Status](https://readthedocs.org/projects/mace/badge/)](https://mace-docs.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/505964914.svg)](https://doi.org/10.5281/zenodo.14103332)

## Table of contents

- [MACE](#mace)
  - [Table of contents](#table-of-contents)
  - [About MACE](#about-mace)
  - [Documentation](#documentation)
  - [Installation](#installation)
    - [pip installation](#installation-from-pypi)
    - [pip installation from source](#installation-from-source)
  - [Usage](#usage)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [XDM Atomic Coefficient Prediction](#xdm-atomic-coefficient-prediction)
  - [Tutorials](#tutorials)
  - [CUDA acceleration with cuEquivariance](#cuda-acceleration-with-cuequivariance)
  - [Weights and Biases for experiment tracking](#weights-and-biases-for-experiment-tracking)
  - [Pretrained Foundation Models](#pretrained-foundation-models)
    - [MACE-MP: Materials Project Force Fields](#mace-mp-materials-project-force-fields)
      - [Example usage in ASE](#example-usage-in-ase)
    - [MACE-OFF: Transferable Organic Force Fields](#mace-off-transferable-organic-force-fields)
      - [Example usage in ASE](#example-usage-in-ase-1)
    - [MACE-Polar: Electrostatics foundation models](#mace-polar-electrostatics-foundation-models)
    - [Finetuning foundation models](#finetuning-foundation-models)
    - [Latest recommended foundation models](#latest-recommended-foundation-models)
  - [Caching](#caching)
  - [Development](#development)
  - [References](#references)
  - [Contact](#contact)
  - [License](#license)

## About MACE

MACE provides fast and accurate machine learning interatomic potentials with higher order equivariant message passing.

This repository contains the MACE reference implementation developed by
Ilyes Batatia, Gregor Simm, David Kovacs, and the group of Gabor Csanyi, and friends (see Contributors).

This fork (MACE-XDM) additionally vendors an extension for predicting per-atom
XDM dispersion coefficients (M1, M2, M3, Veff); see
[XDM Atomic Coefficient Prediction](#xdm-atomic-coefficient-prediction) below.

Also available:

- [MACE in JAX](https://github.com/ACEsuit/mace-jax), currently about 2x times faster at evaluation, but training is recommended in Pytorch for optimal performances.
- [MACE layers](https://github.com/ACEsuit/mace-layer) for constructing higher order equivariant graph neural networks for arbitrary 3D point clouds.

## Documentation

A partial documentation is available at: https://mace-docs.readthedocs.io

## Installation

### 1. Requirements

- Python >= 3.9
- [PyTorch](https://pytorch.org/) >= 1.12 **(training with float64 is not supported with PyTorch 2.1 but is supported with 2.2 and later, Pytorch 2.4.1 is not supported)**

**Make sure to install PyTorch.** Please refer to the [official PyTorch installation](https://pytorch.org/get-started/locally/) for the installation instructions. Select the appropriate options for your system.

### Installation from PyPI

This is the recommended way to install MACE.

```sh
pip install --upgrade pip
pip install mace-torch
```

**Note:** The homonymous package on [PyPI](https://pypi.org/project/MACE/) has nothing to do with this one.

### Installation from source

```sh
git clone https://github.com/ACEsuit/mace.git
pip install ./mace
```

## Usage

### Training

To train a MACE model, you can use the `mace_run_train` script, which should be in the usual place that pip places binaries (or you can explicitly run `python3 <path_to_cloned_dir>/mace/cli/run_train.py`)

```sh
mace_run_train \
    --name="MACE_model" \
    --train_file="train.xyz" \
    --valid_fraction=0.05 \
    --test_file="test.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --E0s='{1:-13.663181292231226, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.0 \
    --batch_size=10 \
    --max_num_epochs=1500 \
    --stage_two \
    --start_stage_two=1200 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
```

To give a specific validation set, use the argument `--valid_file`. To set a larger batch size for evaluating the validation set, specify `--valid_batch_size`.

To control the model's size, you need to change `--hidden_irreps`. For most applications, the recommended default model size is `--hidden_irreps='256x0e'` (meaning 256 invariant messages) or `--hidden_irreps='128x0e + 128x1o'`. If the model is not accurate enough, you can include higher order features, e.g., `128x0e + 128x1o + 128x2e`, or increase the number of channels to `256`. It is also possible to specify the model using the     `--num_channels=128` and `--max_L=1`keys.

It is usually preferred to add the isolated atoms to the training set, rather than reading in their energies through the command line like in the example above. To label them in the training set, set `config_type=IsolatedAtom` in their info fields. 

When training a model from scratch, if you prefer not to use or do not know the energies of the isolated atoms, you can use the option `--E0s="average"` which estimates the atomic energies using least squares regression. Note that using fitted E0s corresponds to fitting the deviations of the atomic energies from the average, rather than fitting the atomization energy (which is the case when using isolated-atom E0s), and this will most likely result in less stable potentials for molecular dynamics applications.

When finetuning foundation models, you can use `--E0s="estimated"`, which estimates the atomic reference energies by solving a linear system that optimally corrects the foundation model's predictions on the training data. This approach computes E0 corrections by first running the foundation model on all training configurations, computing the prediction errors (reference energies minus predicted energies), and then solving a least-squares system to find optimal E0 corrections for each element. This is preferable in general over the 'average' option. 

If the keyword `--stage_two` (previously called swa) is enabled, the energy weight of the loss is increased for the last ~20% of the training epochs (from `--start_stage_two` epochs). This setting usually helps lower the energy errors.

The precision can be changed using the keyword `--default_dtype`, the default is `float64` but `float32` gives a significant speed-up (usually a factor of x2 in training).

The keywords `--batch_size` and `--max_num_epochs` should be adapted based on the size of the training set. The batch size should be increased when the number of training data increases, and the number of epochs should be decreased. An heuristic for initial settings, is to consider the number of gradient update constant to 200 000, which can be computed as $\text{max-num-epochs}*\frac{\text{num-configs-training}}{\text{batch-size}}$.

The code can handle training set with heterogeneous labels, for example containing both bulk structures with stress and isolated molecules. In this example, to make the code ignore stress on molecules, append to your molecules configuration a `config_stress_weight = 0.0`.

By default, a figure displaying the progression of loss and RMSEs during training, along with a scatter plot of the model's inferences on the train, validation, and test sets, will be generated in the results folder at the end of training. This can be disabled using `--plot False`. To track these metrics throughout training (excluding inference on the test set), you can enable periodic plotting for the train and validation sets by specifying `--plot_frequency N`, which updates the plots every Nth epoch.

#### Apple Silicon GPU acceleration

To use Apple Silicon GPU acceleration make sure to install the latest PyTorch version and specify `--device=mps`.

#### Multi-GPU training

For multi-GPU training, use the `--distributed` flag. This will use PyTorch's DistributedDataParallel module to train the model on multiple GPUs. Combine with on-line data loading for large datasets (see below). An example slurm script can be found in `mace/scripts/distributed_example.sbatch`.

#### YAML configuration

Option to parse all or some arguments using a YAML is available. For example, to train a model using the arguments above, you can create a YAML file `your_configs.yaml` with the following content:

```yaml
name: nacl
seed: 2024
train_file: train.xyz
stage_two: yes
start_stage_two: 1200
max_num_epochs: 1500
device: cpu
test_file: test.xyz
E0s:
  41: -1029.2809654211628
  38: -1484.1187695035828
  8: -2042.0330099956639
config_type_weights:
  Default: 1.0

```

And append to the command line `--config="your_configs.yaml"`. Any argument specified in the command line will overwrite the one in the YAML file.

### Evaluation

To evaluate your MACE model on an XYZ file, run the `mace_eval_configs`:

```sh
mace_eval_configs \
    --configs="your_configs.xyz" \
    --model="your_model.model" \
    --output="./your_output.xyz"
```

## XDM Atomic Coefficient Prediction

This fork adds `AtomicXDMMACE`, a MACE variant that predicts per-atom
[XDM](https://en.wikipedia.org/wiki/Van_der_Waals_correction#Exchange-hole_dipole_moment_model)
dispersion coefficients — the moments M1, M2, M3 and the effective volume
Veff — directly from atomic structure, using the same equivariant
message-passing body as MACE's energy models (following the ANI symmetry-function
approach used in [MLXDM](https://github.com/RowleyGroup/MLXDM), but with MACE's
higher-order equivariant features in place of ANI's radial/angular symmetry
functions). Because M1/M2/M3/Veff are invariant per-atom scalars (not an
extensive total like energy), the model reads out `num_xdm_targets` (default 4)
`0e` scalars per atom instead of a single energy.

The equivariant interaction/product-basis body (and every readout but the
last) is shared across all elements, the same as base MACE's energy model --
element identity is injected via the one-hot `node_attrs` at several points
(embedding, skip connections, per-element symmetric-contraction weights), but
the underlying geometric feature extraction is common to every element,
letting rare elements benefit from what abundant ones teach the shared
backbone. The *final* readout, however, is per-element
(`PerElementLinearReadoutBlock`/`PerElementNonLinearReadoutBlock`): each
element gets its own dedicated weight matrix at the last layer rather than
sharing one across all elements. This matters because that last readout is
fit as a flat average over every atom -- a rare element (a small fraction of
total atoms) has little influence on a *shared* final layer relative to
abundant ones, so its fit can be limited by, or even regress toward, whatever
compromise best suits the abundant elements as training progresses. A
per-element final layer removes that competition for the last layer's
capacity specifically, without giving up the shared backbone's benefit for
everything upstream of it.

Predictions are made in per-element standardized (z-score) units: for each
chemical element, the network predicts a standardized residual, which is
mapped back to physical units by `AtomicElementReferenceBlock` via
`physical = mean[Z] + std[Z] * standardized` — the same convention as MLXDM's
own `Shifter` module (`b0 + b1 * x`, with `b0` the per-element mean and `b1` a
per-element standardization width). By default (`--element_stats=mlxdm_2x`),
these mean/width values are *not* computed from your training data — they're
the fixed reference values for H, C, N, O, S, F, Cl taken directly from
RowleyGroup/MLXDM's ANI-2x dispersion model
(`torchanipbe0/resources/dispersion_2x/{m1,m2,m3,v}/best.param`), so a model
trained this way standardizes targets exactly the way MLXDM does. Note that
`b1` is a width chosen to cover MLXDM's own multi-modal, chemically diverse
ANI-2x training distribution (the same element takes on very different values
in different bonding environments), not the literal empirical std of any
particular dataset — a narrower or more homogeneous training set can have a
substantially smaller true std without anything being miscalibrated; use
`mace_compare_xdm_stats` (below) to check. Pass `--element_stats=dataset` to
instead compute mean/std directly from your own training molecules (required
if your dataset includes elements outside that set of 7, and generally gives
a better-scaled gradient signal if your data doesn't span MLXDM's full
chemical breadth).

`--loss_weights W_M1 W_M2 W_M3 W_VEFF` sets the per-property weight in the
standardized-space MSE loss, default `1.0 0.15 0.4 1.0`. This reflects each
moment's role in the XDM dispersion series for nearest-neighbour
intermolecular interactions: C6 (~60% of the energy) depends only on M1 and
Veff (via the atom-in-molecule polarizability), and both also appear in the
numerator and denominator of C8 and C10, so they gate the whole series; M3
additionally enters C8 (~30%) and C10 (~10%); M2 only enters C10, and even
there alongside M3, with no M4 term involved. Pass explicit weights (e.g.
`1 1 1 1` for equal weighting) if this dispersion regime or breakdown doesn't
apply to your use case.

### Dataset format

Training data is expected as one or more ANI-style HDF5 files, e.g. many files
each covering a batch of molecules (as produced by successive active-learning
rounds). Per-molecule groups may be nested under an arbitrary wrapper group
(e.g. a batch/run name) — the actual leaf groups (the ones holding datasets)
are discovered automatically at any depth:

```
/<wrapper>/<molecule>/atomic_numbers   [n_atoms]                          (uint8/int)
/<wrapper>/<molecule>/coordinates      [n_conf, n_atoms, 3]               (Angstrom)
/<wrapper>/<molecule>/M1               [n_conf, n_atoms]
/<wrapper>/<molecule>/M2               [n_conf, n_atoms]
/<wrapper>/<molecule>/M3               [n_conf, n_atoms]
/<wrapper>/<molecule>/Veff             [n_conf, n_atoms]
```

`<wrapper>` may be absent entirely (molecule groups directly at the file's
top level) or nested to any depth — both are handled the same way. The species
key defaults to `atomic_numbers` (an integer array); a symbol array (e.g.
`species`, with entries like `b"H"`, `b"C"`) is also supported via
`--species_key=species`. Key names (`coordinates`, `M1`, `M2`, `M3`, `Veff`)
can be overridden with `--coordinates_key` and `--target_keys` if your files
use different names.

If the same molecule name appears in more than one file (e.g. each file
contributes one new conformer per molecule from a round of active learning),
all of their conformers are pooled together under that molecule name.

### Training

Pass one or more files and/or glob patterns via `--train_files` — either many
per-batch files (`"data/pbe0xdm-ani2x_*.hdf5"`) or a single merged master
file:

```sh
mace_run_train_xdm \
    --train_files "deshaw_350k_pbe0xdm.hdf5" \
    --valid_fraction=0.1 \
    --test_fraction=0.1 \
    --r_max=5.0 \
    --hidden_irreps="128x0e + 128x1o" \
    --num_interactions=2 \
    --batch_size=32 \
    --max_num_epochs=200 \
    --name="xdm_model"
```

`--train_files` is split three ways **by molecule identity** (not individual
conformers): `--valid_fraction` and `--test_fraction` of the molecule names
are held out for validation and testing respectively, so no molecule's
conformers are ever split across train/valid/test, even when the same
molecule's conformers are spread across several files. Validation drives
model selection and early stopping during training; the test set is only
touched once, after training, using the best checkpoint. Pass
`--valid_files`/`--test_files` instead to use separate files for either split
rather than carving them out of `--train_files`.

The resulting split (molecule names per set) is saved to
`<results_dir>/<name>_split.json`, and final test-set MAE/RMSE per property to
`<results_dir>/<name>_test_metrics.json`. Per-element statistics used for
standardization (see above) are stored in the checkpoint regardless of source.
The best model (lowest validation loss) is saved to `<model_dir>/<name>.model`,
ready for evaluation or downstream use.

Pass `--restart_latest` to resume from `<checkpoints_dir>/<name>_latest.pt` --
useful when a job gets interrupted (SLURM walltime, a node failure) partway
through. This restores the `ReduceLROnPlateau` scheduler's and early-stopping's
internal state (how many evaluations since the last improvement) along with
the model/optimizer, so a run that gets restarted repeatedly still decays its
learning rate and early-stops on the same schedule as one that never was --
restarting doesn't reset either patience clock back to zero. Checkpoints
written before this was tracked restart with both clocks starting fresh
(logged as a warning) rather than failing to load.

### Evaluation

To run a trained model on new structures (any ASE-readable format) and
write the predicted per-atom coefficients back out as extended XYZ arrays:

```sh
mace_eval_xdm \
    --model="xdm_model.model" \
    --configs="your_configs.xyz" \
    --output="your_output.xyz"
```

`--model` accepts either the full saved model (`<model_dir>/<name>.model`) or
a training checkpoint (`<checkpoints_dir>/<name>_{latest,best}.pt`) --
whichever you have on hand; both are detected automatically
(`mace.modules.load_xdm_model`). Loading a full model on a CPU-only machine
after training on a GPU also works out of the box, working around an
e3nn<=0.4.x quirk where its internal `torch.jit.load` call doesn't forward
`map_location` (see `mace.tools.safe_jit_load_map_location`/`load_full_model`).

### Checking whether a model (including one still training) is any good

`mace_test_xdm` evaluates a checkpoint on held-out data and reports overall
and per-element MAE/RMSE/R² for each property, a comparison against the
naive "always predict the fixed per-element reference mean" baseline (a
`skill` near 0 means the network isn't beating that prior; closer to 1 is
better), and a predicted-vs-true scatter plot -- useful for sanity-checking
a model whose training hasn't finished yet, not just the final one:

```sh
mace_test_xdm \
    --model="checkpoints/xdm_model_best.pt" \
    --test_files="your_xdm_dataset.h5" \
    --split_file="results/xdm_model_split.json" \
    --output_dir="test_results"
```

`--split_file` (the file `mace_run_train_xdm` writes automatically) restricts
evaluation to exactly the molecules held out as the test set, so results
reflect true generalization rather than partly re-scoring training data;
omitting it evaluates every molecule in `--test_files` instead. Add
`--max_conformers 20000` to get a quick read on a very large dataset without
waiting to score the whole thing.

`mace_test_xdm`'s metrics are on the raw M1/M2/M3/Veff regression targets,
which don't by themselves say how much any given error actually matters
physically -- a mediocre M2 fit may barely move the dispersion energy, while
a small M1 error can shift it substantially. `mace_test_xdm_dispersion`
instead judges the model by the actual downstream quantity: it computes the
XDM dispersion energy (via the same Becke-Johnson-damped C6+C8+C10 formula
used in training/production) from both the model's predicted per-atom
moments and the dataset's true ones, per test molecule, and reports MAE/RMSE/
R²/Pearson r for the total energy and for the C6, C8, C10 components
separately -- so it's clear whether error is concentrated in the term that
matters most (C6, typically ~60% of nearest-neighbour intermolecular
dispersion) or in a smaller one:

```sh
mace_test_xdm_dispersion \
    --model="checkpoints/xdm_model_best.pt" \
    --test_files="your_xdm_dataset.h5" \
    --split_file="results/xdm_model_split.json" \
    --output_dir="test_dispersion_results"
```

Same `--split_file`/`--max_conformers` conventions as `mace_test_xdm`. Since
the dispersion energy needs an O(atoms²) pairwise distance matrix per
molecule (not just per-atom predictions), `--batch_size` defaults lower (32).

If training looks slower or noisier than expected, it's worth checking
whether the *default* fixed MLXDM reference mean/standardization-width (used
to standardize targets, see below) actually matches your dataset's real
distribution -- `mace_compare_xdm_stats` computes the actual empirical
per-element mean/std from your data and prints it side by side with the fixed
values, flagging any element/property where they disagree by more than 1
empirical std (mean) or 2x (width). A width mismatch on its own isn't
necessarily wrong (MLXDM's width covers a broader, multi-modal chemical space
than any one dataset may sample), but it does mean the standardized loss is
harder to interpret and training gets a weaker gradient signal than a width
matched to your own data would give:

```sh
mace_compare_xdm_stats \
    --train_files="your_xdm_dataset.h5" \
    --split_file="results/xdm_model_split.json" \
    --atomic_numbers="1,6,7,8,9,16,17"
```

A mismatch isn't a code bug, but it does mean the network first has to
compensate for a miscalibrated prior before it's really fitting structure --
a large std mismatch in particular can produce unnecessarily large or small
standardized-space gradients. `--max_molecules` (default 2000) subsamples for
speed, since the underlying statistics computation isn't vectorized and
scanning millions of conformers would take a long time.

### Adding XDM dispersion energy to a short-range MACE potential

`MACEXDMDispersion` (`mace.modules`) combines a short-range MACE energy model
(trained on dispersion-deficient reference energies, e.g. bare PBE0) with a
trained `AtomicXDMMACE` to give a total potential:

```
E_total = E_short_range(structure) + E_dispersion(XDM)
```

This mirrors RowleyGroup/MLXDM's own `ANIDispersion` module, which is a plain
sum of an ANI backbone's energy and a dispersion model's energy -- the
standard DFT+D-style recipe. The two sub-models are trained completely
independently and may use different element orderings; `MACEXDMDispersion`
reconciles them automatically.

The dispersion energy itself (`XDMDispersionEnergy`) implements the
Becke-Johnson-damped XDM combining rules exactly as used by MLXDM's ANI-2x
dispersion model, transcribed from its source
(`torchanipbe0/dispersion/nn.py`, `torchanipbe0/models.py`):

```
alpha_A   = Veff_A * alpha_free[Z_A] / v_free[Z_A]
C6_AB     = M1_A*M1_B / (M1_A/alpha_A + M1_B/alpha_B)
C8_AB     = 1.5*(M1_A*M2_B + M1_B*M2_A) / (M1_A/alpha_A + M1_B/alpha_B)
C10_AB    = 2*(M1_A*M3_B + M3_A*M1_B + 2.1*M2_A*M2_B) / (M1_A/alpha_A + M1_B/alpha_B)
R_crit,AB = (sqrt(C8/C6) + (C10/C6)^0.25 + sqrt(C10/C8)) / 3        (bohr)
R_vdw,AB  = a2 + a1 * R_crit,AB * 0.529177249                        (Angstrom)
E_disp    = -sum_pairs sum_{n=6,8,10}  C_n,AB / (r_AB^n + R_vdw,AB^n) * 0.529177249^n
```

with `alpha_free`/`v_free` (free-atom polarizability/volume) and the
damping parameters `a1=0.4186, a2=2.6791` defaulting to MLXDM's fixed
ANI-2x/PBE0-XDM values for H, C, N, O, S, F, Cl
(`mace.data.mlxdm_2x_polarizability_reference`). The dispersion sum uses a
dense pairwise distance matrix over whole (finite, non-periodic) molecules
rather than a fixed-radius neighbor list, masked by a 14 Angstrom cutoff by
default -- appropriate since dispersion decays slowly and this cutoff is much
larger than a typical short-range MACE cutoff.

Use the combined potential as an ASE calculator:

```python
from ase import Atoms
from mace.calculators import MACEXDMDispersionCalculator

calc = MACEXDMDispersionCalculator(
    short_range_model_path="short_range_pbe0.model",
    xdm_model_path="xdm_model.model",
    device="cpu",
)
atoms = Atoms(...)
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

Forces are computed by autograd through the whole combined energy (both
sub-models' contributions), not by adding separately-computed forces, so
they are exact for the combined potential. Currently supports finite
molecules only (no PBC/stress).

#### Evaluating on intermolecular complex/monomer data

`mace_eval_intermolecular_xdm` judges the combined potential on what it's
actually for: predicted intermolecular interaction energies, `E(complex) -
E(frag_1) - E(frag_2)`, against QM reference values. It expects an HDF5 file
with one leaf group per complex and one per monomer fragment -- e.g. as
written by a postg/Gaussian-parsing script that computes each complex and its
two fragments separately (the reference PBE0 energy and XDM dispersion energy
each stored per structure). Complex/fragment groups are matched by their
trailing `_<index>` (fragments additionally end `.frag_1`/`.frag_2`) --
deliberately independent of whatever prefix text comes before the index, so a
complex/fragment prefix mismatch in the naming (not uncommon -- data-prep
scripts evolve) doesn't silently drop triplets:

```sh
mace_eval_intermolecular_xdm \
    --short_range_model="short_range_pbe0.model" \
    --xdm_model="xdm_model.model" \
    --data_files="complexes_and_monomers.h5" \
    --output_dir="intermolecular_results"
```

Reports MAE/RMSE/R²/Pearson r (in kcal/mol) for three interaction energies
separately: PBE0 alone (short-range model only, no dispersion), XDM alone
(the dispersion-energy output only), and PBE0+XDM (the full combined
model) -- matching the three quantities a typical postg-based reference
pipeline computes. `--energy_key`/`--exdm_key` (default `energies`/`e_xdm`)
name the QM reference Hartree-unit datasets on each structure;
`--max_triplets` subsamples complexes for a quick look. Writes a JSON report,
a CSV of every complex's true/predicted values, and a scatter plot
(`--no_plots` to skip).

Aggregate metrics on a large, diverse complex/monomer dataset can be
dominated by a small number of catastrophic outliers rather than reflecting
typical performance -- e.g. a short-range model trained on organic-molecule
data (such as ANI-2x) has essentially no exposure to a bare, isolated
diatomic like H2 as a standalone fragment, and can mispredict its energy by
100+ kcal/mol, swamping the RMSE/R² for the other 99.8% of complexes it
handles well. `--min_fragment_atoms N` skips any triplet whose complex,
frag_1, or frag_2 has fewer than N atoms (e.g. `--min_fragment_atoms 3`
excludes diatomics like H2); `--skip_homonuclear_fragments` more
specifically skips any triplet where a structure consists of a single
element only (H2, O2, N2, Cl2, ...) regardless of size. Both are opt-in
(default: no filtering) so the unfiltered numbers -- including whatever such
outliers are actually in the data -- are always available too.

## Tutorials

You can run our [Colab tutorial](https://colab.research.google.com/drive/1D6EtMUjQPey_GkuxUAbPgld6_9ibIa-V?authuser=1#scrollTo=Z10787RE1N8T) to quickly get started with MACE.

We also have a more detailed Colab tutorials on:

- [Introduction to MACE training and evaluation](https://colab.research.google.com/drive/1ZrTuTvavXiCxTFyjBV4GqlARxgFwYAtX)
- [Introduction to MACE active learning and fine-tuning](https://colab.research.google.com/drive/1oCSVfMhWrqHTeHbKgUSQN9hTKxLzoNyb)
- [MACE theory and code (advanced)](https://colab.research.google.com/drive/1AlfjQETV_jZ0JQnV5M3FGwAM2SGCl2aU)

## CUDA acceleration with cuEquivariance

MACE supports CUDA acceleration with the cuEquivariance library. To install the library and use the acceleration, see our documentation at https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html.

## On-line data loading for large datasets

If you have a large dataset that might not fit into the GPU memory it is recommended to preprocess the data on a CPU and use on-line dataloading for training the model. To preprocess your dataset specified as an xyz file run the `preprocess_data.py` script. An example is given here:

```sh
mkdir processed_data
python ./mace/scripts/preprocess_data.py \
    --train_file="/path/to/train_large.xyz" \
    --valid_fraction=0.05 \
    --test_file="/path/to/test_large.xyz" \
    --atomic_numbers="[1, 6, 7, 8, 9, 15, 16, 17, 35, 53]" \
    --r_max=4.5 \
    --h5_prefix="processed_data/" \
    --compute_statistics \
    --E0s="average" \
    --seed=123 \
```

To see all options and a little description of them run `python ./mace/scripts/preprocess_data.py --help` . The script will create a number of HDF5 files in the `processed_data` folder which can be used for training. There will be one folder for training, one for validation and a separate one for each `config_type` in the test set. To train the model use the `run_train.py` script as follows:

```sh
python ./mace/scripts/run_train.py \
    --name="MACE_on_big_data" \
    --num_workers=16 \
    --train_file="./processed_data/train.h5" \
    --valid_file="./processed_data/valid.h5" \
    --test_dir="./processed_data" \
    --statistics_file="./processed_data/statistics.json" \
    --model="ScaleShiftMACE" \
    --num_interactions=2 \
    --num_channels=128 \
    --max_L=1 \
    --correlation=3 \
    --batch_size=32 \
    --valid_batch_size=32 \
    --max_num_epochs=100 \
    --stage_two \
    --start_stage_two=60 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --error_table='PerAtomMAE' \
    --device=cuda \
    --seed=123 \
```

## Weights and Biases for experiment tracking

If you would like to use MACE with Weights and Biases to log your experiments simply install with

```sh
pip install ./mace[wandb]
```

And specify the necessary keyword arguments (`--wandb`, `--wandb_project`, `--wandb_entity`, `--wandb_name`, `--wandb_log_hypers`)

## Pretrained Foundation Models

We provide a series of pretrained foundation models for various applications. These models can be used directly for inference, or as a starting point for fine-tuning on a new dataset.
Foundation models are a rapidly evolving field. Please look at the [MACE-MP GitHub repository](https://github.com/ACEsuit/mace-foundations/releases) and the [MACE-OFF23 GitHub repository](https://github.com/ACEsuit/mace-off/releases) for the latest releases.

### Latest Recommended Foundation Models

| Model Name           | Elements Covered | Training Dataset | Level of Theory     | Target System     | Model Size                                                                                                                                                                                                                                                                                                                                                                        | GitHub Release | Notes                                                              | License |
| -------------------- | ---------------- | ---------------- | ------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------ | ------- |
| MACE-MP-0a           | 89               | MPTrj            | DFT (PBE+U)         | Materials         | [small](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model), [medium](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model), [large](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model) | >=v0.3.6       | Initial release of foundation model.                               | MIT     |
| MACE-MP-0b3          | 89               | MPTrj            | DFT (PBE+U)         | Materials         | [medium](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model)                                                                                                                                                                                                                                                                      | >=v0.3.10      | Improved high pressure stability and reference energies.           | MIT     |
| MACE-MPA-0           | 89               | MPTrj + sAlex    | DFT (PBE+U)         | Materials         | [medium-mpa-0](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model)                                                                                                                                                                                                                                                                  | >=v0.3.10      | Improved accuracy for materials, improved high pressure stability. | MIT     |
| MACE-OMAT-0          | 89               | OMAT             | DFT (PBE+U) VASP 54 | Materials         | [medium-omat-0](https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model)                                                                                                                                                                                                                                                               | >=v0.3.10      |                                                                    | ASL     |
| MACE-OFF23           | 10               | SPICE v1         | DFT (wB97M+D3)      | Organic Chemistry | [small](https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model), [medium](https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_medium.model), [large](https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_large.model)                                                                                                    | >=v0.3.6       | Initial release covering neutral organic chemistry.                | ASL     |
| MACE-MATPES-PBE-0    | 89               | MATPES-PBE       | DFT (PBE)           | Materials         | [medium](https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model)                                                                                                                                                                                                                                                               | >=v0.3.10      | No +U correction.                                                  | ASL     |
| MACE-MATPES-r2SCAN-0 | 89               | MATPES-r2SCAN    | DFT (r2SCAN)        | Materials         | [medium](https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model)                                                                                                                                                                                                                                                            | >=v0.3.10      | Better functional for materials.                                   | ASL     |
| MACE-OMOL-0 | 89               | OMOL    | DFT (wB97M-VV10)        | Molecules/Transition metals/Cations         | [large](https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/MACE-omol-0-extra-large-1024.model)                                                                                                                                                                                                                                                           | >=v0.3.14      | Charge/Spin embedding, very good molecular accuracy.                                   | ASL     |
| MACE-MH-0/1 | 89               | OMAT/OMOL/OC20/MATPES    | DFT (PBE/R2SCAN/wB97M-VV10)        | Inorganic crystals, molecules and surfaces. [More info.](https://huggingface.co/mace-foundations/mace-mh-1)         | [mh-0](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model) [mh-1](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model)                                                                                                                                                                                                                                                           | >=v0.3.14      | Very good cross domain performance on surfaces/bulk/molecules.   | ASL     |


### MACE-MP: Materials Project Force Fields

We have collaborated with the Materials Project (MP) to train a universal MACE potential covering 89 elements on 1.6 M bulk crystals in the [MPTrj dataset](https://figshare.com/articles/dataset/23713842) selected from MP relaxation trajectories.
The models are releaed on GitHub at https://github.com/ACEsuit/mace-foundations.
If you use them please cite [our paper](https://arxiv.org/abs/2401.00096) which also contains an large range of example applications and benchmarks.

> [!CAUTION]
> The MACE-MP models are trained on MPTrj raw DFT energies from VASP outputs, and are not directly comparable to the MP's DFT energies or CHGNet's energies, which have been applied MP2020Compatibility corrections for some transition metal oxides, fluorides (GGA/GGA+U mixing corrections), and 14 anions species (anion corrections). For more details, please refer to the [MP Documentation](https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/thermodynamic-stability/anion-and-gga-gga+u-mixing) and [MP2020Compatibility.yaml](https://github.com/materialsproject/pymatgen/blob/master/pymatgen/entries/MP2020Compatibility.yaml).

#### Example usage in ASE

```py
from mace.calculators import mace_mp
from ase import build

atoms = build.molecule('H2O')
calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device='cuda')
atoms.calc = calc
print(atoms.get_potential_energy())
```

### MACE-OFF: Transferable Organic Force Fields

There is a series (small, medium, large) transferable organic force fields. These can be used for the simulation of organic molecules, crystals and molecular liquids, or as a starting point for fine-tuning on a new dataset. The models are released under the [ASL license](https://github.com/gabor1/ASL).
The models are releaed on GitHub at https://github.com/ACEsuit/mace-off.
If you use them please cite [our paper](https://arxiv.org/abs/2312.15211) which also contains detailed benchmarks and example applications.

#### Example usage in ASE

```py
from mace.calculators import mace_off
from ase import build

atoms = build.molecule('H2O')
calc = mace_off(model="medium", device='cuda')
atoms.calc = calc
print(atoms.get_potential_energy())
```

### MACE-Polar: Electrostatics foundation models

PolarMACE checkpoints are electrostatics foundation models for molecular chemistry, trained on the OMol25 dataset.
For usage, outputs, and training/finetuning details, see the PolarMACE guide:

- https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html

### Finetuning foundation models

To finetune one of the mace-mp-0 foundation model, you can use the `mace_run_train` script with the extra argument `--foundation_model=model_type`. For example to finetune the small model on a new dataset, you can use:

```sh
mace_run_train \
  --name="MACE" \
  --foundation_model="small" \
  --train_file="train.xyz" \
  --valid_fraction=0.05 \
  --test_file="test.xyz" \
  --energy_weight=1.0 \
  --forces_weight=1.0 \
  --E0s="average" \
  --lr=0.01 \
  --scaling="rms_forces_scaling" \
  --batch_size=2 \
  --max_num_epochs=6 \
  --ema \
  --ema_decay=0.99 \
  --amsgrad \
  --default_dtype="float32" \
  --device=cuda \
  --seed=3
```

Other options are "medium" and "large", or the path to a foundation model.
If you want to finetune another model, the model will be loaded from the path provided `--foundation_model=$path_model`, all the hypers will be extracted automatically.

## Caching

By default automatically downloaded models, like mace_mp, mace_off and data for fine tuning, end up in `~/.cache/mace`. The path can be changed by using
the environment variable XDG_CACHE_HOME. When set, the new cache path expands to $XDG_CACHE_HOME/.cache/mace

## Development

This project uses [pre-commit](https://pre-commit.com/) to execute code formatting and linting on commit.
We also use `black`, `isort`, `pylint`, and `mypy`.
We recommend setting up your development environment by installing the `dev` packages
into your python environment:

```bash
pip install -e ".[dev]"
pre-commit install
```

The second line will initialise `pre-commit` to automaticaly run code checks on commit.
We have CI set up to check this, but we _highly_ recommend that you run those commands
before you commit (and push) to avoid accidentally committing bad code.

We are happy to accept pull requests under an [MIT license](https://choosealicense.com/licenses/mit/). Please copy/paste the license text as a comment into your pull request.

## References

If you use this code, please cite our papers:

```bibtex
@inproceedings{Batatia2022mace,
  title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Ilyes Batatia and David Peter Kovacs and Gregor N. C. Simm and Christoph Ortner and Gabor Csanyi},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=YPpSngE-ZU}
}

@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2205.06643},
  eprint = {2205.06643},
  eprinttype = {arxiv},
  doi = {10.48550/arXiv.2205.06643},
  archiveprefix = {arXiv}
 }
```

## Contact

If you have any questions, please contact us at ilyes.batatia@ens-paris-saclay.fr.

For bugs or feature requests, please use [GitHub Issues](https://github.com/ACEsuit/mace/issues).

## License

The MACE code is published and distributed under the [MIT License](MIT.md). (Note that some of the models linked above come with different licenses).
