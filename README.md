# DSC 180A Capstone Project: Optimizer Studies on Heavy-Tailed Distributions

This is my capstone project for DSC 180A. I'm looking at how different optimizers behave when you have heavy-tailed class imbalance (like Zipf distributions) in your data. Basically trying to understand why Adam works better than SGD in these cases.

## What's in here

There are two main parts to this project:

1. **Week 1-5**: Reproducing the paper "Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models". I run longer experiments comparing SGD vs Adam on CNN/ResNet models with heavy-tailed datasets.

2. **Week 6**: Looking at scaling laws for Gradient Descent vs Sign Descent on linear bigram models. This is a different paper about how convergence scales with vocabulary size.

## Project structure

The main notebooks are in `reports/scaling_validation_v3/`:
- `week1-5_research.ipynb` - the long-run SGD vs Adam experiments
- `week6_research.ipynb` - the GD vs Sign Descent scaling stuff
- `dsc180adamvssgd.ipynb` - original reproduction notebook (from earlier weeks)

Plots and results get saved in subdirectories. The notebooks will create the folders automatically.

## Setting up

You'll need Python 3.8 or higher. I used Python 3.9 but 3.8 should work fine too.

### Installing dependencies

Just run:
```bash
pip install -r requirements.txt
```

Or if you want to use conda (which I'd recommend):
```bash
conda create -n dsc180a python=3.9
conda activate dsc180a
pip install -r requirements.txt
```

The main things you need:
- PyTorch (for the neural network stuff)
- numpy, pandas (data stuff)
- matplotlib, seaborn (plotting)
- tqdm (for progress bars - super helpful when things take forever)
- transformers (optional, for Week 6 if you want to use real text data)
- torchvision (optional, for CIFAR10 in Week 1-5)

The optional ones aren't required - the notebooks will just use synthetic data instead if they're not installed.

## Data stuff

### Week 1-5

For the Week 1-5 experiments, I'm using synthetic heavy-tailed datasets that get generated in the notebook. No external data needed! The code creates Zipf distributions on the fly.

If you have torchvision installed, it can also use CIFAR10, but honestly the synthetic data works fine and is faster. CIFAR10 will download automatically to `reports/scaling_validation_v3/data/` if you use it.

### Week 6

Week 6 is supposed to use OpenWebText, but that's a huge dataset and takes forever to download/process. So I included placeholder code for it, but by default it just uses synthetic Zipfian data. 

If you really want to use OpenWebText:
- Download from https://skylion007.github.io/OpenWebTextCorpus/
- Or use HuggingFace: `datasets.load_dataset("openwebtext")`
- Put it in `reports/scaling_validation_v3/data/openwebtext/`

But honestly, the synthetic data works fine for reproducing the main results. The paper's findings hold up with synthetic data too.

## Running the notebooks

### Week 1-5 notebook

Open `reports/scaling_validation_v3/week1-5_research.ipynb` in Jupyter and just run all cells. It's pretty straightforward.

**Important:** There's a `FAST_RUN` flag at the top. If you set it to `True`, it runs smaller experiments (12 epochs, 4K samples) which takes like 10-15 minutes. With `False`, it does the full thing (36 epochs, 20K samples) which takes about 90 minutes on CPU or 20-30 minutes if you have a GPU.

The notebook will:
- Generate synthetic heavy-tailed datasets
- Train TinyCNN and TinyResNet with SGD+Momentum and AdamW
- Save plots to `week1-5_plots/`
- Save results to `week1-5_long_run_summaries.csv`

### Week 6 notebook

Open `reports/scaling_validation_v3/week6_research.ipynb` and run all cells.

This one implements the closed-form GD dynamics and simulates Sign Descent. It's mostly synthetic data by default, which is fine. Takes maybe 15-30 minutes to run.

You can adjust:
- `SAMPLE_SIZE` - how many tokens (default 100000)
- `VOCAB_SIZE` - vocabulary size (default 10000)
- `USE_PRECOMPUTED` - if you have pre-computed stats saved somewhere

Plots get saved to `week6_plots/` with names like `week6_experiment1_gd_vocab_scaling.png` etc.

## Reproducibility

Both notebooks use `SEED = 2025` for random number generation, so you should get the same results (or at least very close) if you run them multiple times.

For hardware, you'll want:
- At least 8GB RAM (16GB is better)
- CPU is fine, but GPU speeds things up a lot for Week 1-5
- Maybe 2GB free space for dependencies and outputs

## Common issues

**GPU not working?** The notebooks automatically fall back to CPU, so it'll just be slower. No big deal.

**Out of memory?** Try:
- Setting `FAST_RUN = True` in Week 1-5
- Reducing `VOCAB_SIZE` in Week 6 (like 5000 instead of 10000)
- Reducing `SAMPLE_SIZE` in Week 6

**Missing transformers/torchvision?** That's fine - the notebooks will just use simpler methods or synthetic data. Everything still works.

**Progress bars not showing?** Make sure tqdm is installed. Sometimes Jupyter needs a restart if you install it mid-session.

## Output files

After running, you'll get:

**Week 1-5:**
- `week1-5_plots/week1-5_val_accuracy_curves.png`
- `week1-5_plots/week1-5_val_loss_curves.png`
- `week1-5_plots/week1-5_final_metrics.png`
- `week1-5_long_run_summaries.csv` (has all the experiment results)

**Week 6:**
- Various plots in `week6_plots/` showing the scaling law experiments

## Papers

If you use this code, please cite:
1. Kunstner, F., Yadav, P., Milligan, K., Schmidt, M., & Bietti, A. (2024). Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models.

2. [The scaling laws paper - I'll add the full citation later]

## Questions?

If something doesn't work or you have questions, feel free to open an issue on GitHub or reach out.
