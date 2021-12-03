# Re-implementation of the paper 'Grokking: Generalization beyond overfitting on small algorithmic datasets'

## Paper
Original paper can be found [here](https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf)

## Datasets
I'm not super clear on how they defined their division. I am using integer division:
- $$x\circ y = (x // y) mod p$$, for some prime $$p$$ and $$0\leq x,y \leq p$$
- $$x\circ y = (x // y) mod p$$ if y is odd else (x - y) mod p, for some prime $$p$$ and $$0\leq x,y \leq p$$

## Hyperparameters
The default hyperparameters are from the paper, but can be adjusted via the command line when running `train.py`

## Running experiments
To run with default settings, simply run `python train.py`.
The first time you train on any dataset you have to specify `--force_data`.

### Arguments:
### optimizer args
- "--lr", type=float, default=1e-3
- "--weight_decay", type=float, default=1
- "--beta1", type=float, default=0.9
- "--beta2", type=float, default=0.98

### model args
- "--num_heads", type=int, default=4
- "--layers", type=int, default=2
- "--width", type=int, default=128

### data args
- "--data_name", type=str, default="perm", choices=[
  - "perm_xy", # permutation composition x * y
  - "perm_xyx1", # permutation composition x * y * x^-1
  - "perm_xyx", # permutation composition x * y * x
  - "plus", # x + y
  - "minus", # x - y
  - "div", # x / y
  - "div_odd", # x / y if y is odd else x - y
  - "x2y2", # x^2 + y^2
  - "x2xyy2", # x^2 + y^2 + xy
  - "x2xyy2x", # x^2 + y^2 + xy + x
  - "x3xy", # x^3 + y
  - "x3xy2y" # x^3 + xy^2 + y
]
- "--num_elements", type=int, default=5  (choose 5 for permutation data, 97 for arithmetic data)
- "--data_dir", type=str, default="./data"
- "--force_data", action="store_true", help="Whether to force dataset creation."

### training args
- "--batch_size", type=int, default=512
- "--steps", type=int, default=10**5
- "--train_ratio", type=float, default=0.5
- "--seed", type=int, default=42
- "--verbose", action="store_true"
- "--log_freq", type=int, default=10
- "--num_workers", type=int, default=4