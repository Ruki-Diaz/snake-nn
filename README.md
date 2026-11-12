# snake-nn

A neural network that learns Snake. NumPy only: hand-written forward pass, Q-learning loss, backprop and Adam.

- 11 inputs -> 64 hidden (ReLU) -> 3 outputs, 963 weights
- Trained in ~25 s on CPU: best game 65, greedy average 32.5
- Static site, inference runs in the browser

## Run
    python train.py      # writes data.json (checkpoints + score log)
    python build.py      # embeds data.json into template.html -> index.html
    open index.html

## Deploy ($0)
Push to GitHub, import the repo at vercel.com/new, no build settings needed.
