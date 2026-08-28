# Fashion-MNIST: from-scratch CNN with calibration analysis

This project trains a convolutional neural network to classify Fashion-MNIST images into 10 clothing categories. Its purpose is to demonstrate building, training, and evaluating a CNN in PyTorch on a task where deep learning is genuinely the right tool. As a baseline I fit a logistic regression on the flattened pixels; the CNN beat it by roughly 5 points, quantifying the value of the spatial structure the flattening discards. This is a deliberately small, from-scratch build — the architecture is intentionally minimal and untuned, because the goal was understanding every component and making honest modelling decisions, not chasing benchmark accuracy.

## Approach

**Data.** The dataset is split three ways: a training set the model learns from, a validation set used for all development decisions, and a held-out test set touched only once at the end, to estimate generalisation without leakage.

**Model.** A small CNN: two convolutional blocks (each convolution followed by a ReLU activation and max-pooling), then a linear classification head. Convolution is the right choice here because pixel adjacency is informative — nearby pixels form meaningful local patterns — and convolution exploits that local structure, which a model on flattened pixels cannot.

**Training.** A hand-written training loop, using cross-entropy loss and the Adam optimiser, in batches of 64, on Apple MPS.

**Evaluation.** On held-out data: accuracy, a per-class confusion matrix, and a calibration check (Expected Calibration Error and a reliability diagram) to measure whether the model's confidence is trustworthy.

## Decisions

**CNN over the linear baseline.** As a reference point I fit a logistic regression on the flattened 784-pixel vectors, on the same data split as the CNN. It reached 85.1% validation accuracy against the CNN's 90.1% — a roughly 5-point gap that quantifies the value of the spatial structure the flattening discards and the convolution exploits.

**Shipped without dropout.** Before adding any regulariser, I measured the gap between training and validation accuracy: about 1 percentage point, meaning the model was barely overfitting. I added dropout (p=0.25) as a controlled experiment anyway. It did not improve validation accuracy, and it noticeably reduced recall on the hardest class (shirt, which is visually ambiguous with several other upper-garment classes). This is expected: a regulariser trades training fit for generalisation only when there is overfitting to trade against, and here there was almost none — so dropout cost accuracy on hard cases for no generalisation gain. I shipped the plain model; the `use_dropout` flag keeps the experiment reproducible.

**Measured calibration, applied no correction.** Modern neural networks are documented to be overconfident, so I expected to need calibration. I measured Expected Calibration Error on validation: roughly 0.008, i.e. already well-calibrated. I therefore did not apply temperature scaling — with essentially no miscalibration to correct, a fix would risk degrading the probabilities for no benefit. The final test-set ECE (0.0052) confirmed the model is well-calibrated on data it never influenced.

## Results

On the sealed test set, the final model reaches **89.25% accuracy** with an **Expected Calibration Error of 0.0052** — accurate and well-calibrated, on data used only once for this final measurement.

The per-class confusion matrix shows where the errors fall: they concentrate almost entirely within the visually-similar upper-garment classes — shirt, T-shirt, pullover, and coat — which are genuinely hard to separate at 28×28 greyscale. Shirt is the weakest class, most often confused with T-shirt. Structurally distinct classes (trousers, footwear, bags) are classified near-perfectly. This matches Fashion-MNIST's known difficulty structure rather than being a novel finding — the model fails on the genuinely ambiguous cases and succeeds on the separable ones.

The reliability diagram (`reports/figures/`) and the full confusion matrix are produced by the evaluation run.

## Caveats and limitations

- **Single training run.** All results come from one seed. In particular, the finding that dropout reduced shirt-class recall is from a single run — the magnitude may partly reflect run-to-run variance rather than a robust effect, which repeated seeds would be needed to confirm.
- **Small model on an easy dataset.** The network is deliberately minimal and Fashion-MNIST is a clean, balanced benchmark. The calibration finding is a consequence of this: shallow models on easy data tend to be well-calibrated, so "no calibration needed" should not be read as a general claim about neural networks — a deeper model on harder data would likely need it.
- **Benchmark, not real-world data.** Fashion-MNIST is low-resolution, balanced, and cleanly labelled. Performance here says nothing about messy, imbalanced, or high-resolution real-world image data.
- **No hyperparameter tuning.** Architecture and training settings were fixed by sensible defaults, not searched. The goal was a defensible, understood pipeline, not maximal accuracy — a tuned model would likely score higher.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
```

## Repository structure

```
src/
├── __init__.py        
├── data.py        # dataset loading and train/val/test split
├── model.py       # SmallCNN definition
├── train.py       # training loop, evaluation, main entry point
├── evaluate.py    # accuracy and prediction helpers
├── calibrate.py   # ECE computation and reliability diagram
└── utils.py       # device selection
scripts/
└── baseline.py    # one-off logistic-regression baseline
tests/
└── test_ece.py    # ECE unit tests
reports/figures/   # reliability diagram
```

## Environment

Built and run on Apple Silicon (M1, MPS backend). Falls back to CPU where MPS is unavailable.