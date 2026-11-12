"""Neural network that plays Snake. NumPy only: no PyTorch, no autograd.

State (11 inputs): danger straight/right/left, heading (right/down/left/up),
food left/right/up/down. Actions (3): straight, turn right, turn left.
Learning: Q-learning with replay, hand-written forward pass, backprop and Adam.

Run:  python train.py  ->  data.json (checkpoints + score log)
"""
import json
import random
from collections import deque

import numpy as np

W = H = 20
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up (clockwise)
GAMMA, LR, BATCH = 0.9, 1e-3, 512
N_IN, N_HID, N_OUT = 11, 64, 3


class Snake:
    def __init__(self, rng):
        self.rng = rng
        self.reset()

    def reset(self):
        self.dir = 0
        self.body = [(W // 2, H // 2), (W // 2 - 1, H // 2), (W // 2 - 2, H // 2)]
        self.score = 0
        self.idle = 0
        self._food()
        return self.state()

    def _food(self):
        free = [(x, y) for x in range(W) for y in range(H) if (x, y) not in self.body]
        self.food = self.rng.choice(free)

    def _hit(self, p):
        return not (0 <= p[0] < W and 0 <= p[1] < H) or p in self.body[:-1]

    def _ahead(self, d):
        hx, hy = self.body[0]
        dx, dy = DIRS[d % 4]
        return (hx + dx, hy + dy)

    def state(self):
        hx, hy = self.body[0]
        fx, fy = self.food
        s = [
            self._hit(self._ahead(self.dir)),
            self._hit(self._ahead(self.dir + 1)),
            self._hit(self._ahead(self.dir - 1)),
            self.dir == 0, self.dir == 1, self.dir == 2, self.dir == 3,
            fx < hx, fx > hx, fy < hy, fy > hy,
        ]
        return np.array(s, dtype=np.float32)

    def step(self, action):
        self.dir = (self.dir + (0, 1, -1)[action]) % 4
        head = self._ahead(self.dir)
        self.idle += 1
        if self._hit(head) or self.idle > 100 * len(self.body):
            return self.state(), -10.0, True
        self.body.insert(0, head)
        if head == self.food:
            self.score += 1
            self.idle = 0
            self._food()
            return self.state(), 10.0, False
        self.body.pop()
        return self.state(), 0.0, False


class Net:
    """11 -> 64 (ReLU) -> 3, with Adam. Backprop written by hand."""

    def __init__(self, rng):
        self.p = {
            "W1": rng.normal(0, np.sqrt(2 / N_IN), (N_IN, N_HID)).astype(np.float32),
            "b1": np.zeros(N_HID, np.float32),
            "W2": rng.normal(0, np.sqrt(2 / N_HID), (N_HID, N_OUT)).astype(np.float32),
            "b2": np.zeros(N_OUT, np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x):
        h = np.maximum(0, x @ self.p["W1"] + self.p["b1"])
        return h, h @ self.p["W2"] + self.p["b2"]

    def train(self, x, a, y):
        h, q = self.forward(x)
        n = len(x)
        dq = np.zeros_like(q)
        idx = np.arange(n)
        dq[idx, a] = 2 * (q[idx, a] - y) / n          # dL/dQ = 2(Q - y)
        g = {
            "W2": h.T @ dq,
            "b2": dq.sum(0),
        }
        dh = (dq @ self.p["W2"].T) * (h > 0)           # through ReLU
        g["W1"] = x.T @ dh
        g["b1"] = dh.sum(0)
        self.t += 1
        for k in self.p:                               # Adam
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= LR * mh / (np.sqrt(vh) + 1e-8)

    def export(self):
        return {k: np.round(v, 4).tolist() for k, v in self.p.items()}


def learn(net, batch):
    s, a, r, s2, d = map(np.array, zip(*batch))
    _, q2 = net.forward(s2.astype(np.float32))
    y = r + GAMMA * q2.max(1) * (1 - d)
    net.train(s.astype(np.float32), a, y.astype(np.float32))


def main(games=400, seed=7, checkpoints=(0, 25, 80, 400)):
    rng = random.Random(seed)
    net = Net(np.random.default_rng(seed))
    env = Snake(rng)
    memory = deque(maxlen=100_000)
    scores, ckpts, best = [], [], 0
    best_net = None
    for g in range(games + 1):
        if g in checkpoints and g != games:
            ckpts.append({"game": g, "best": best, "net": net.export()})
        if g == games:
            break
        eps = max(0.0, 0.8 * (1 - g / 90))           # mostly random at first
        s, done = env.reset(), False
        while not done:
            if rng.random() < eps:
                a = rng.randrange(3)
            else:
                a = int(np.argmax(net.forward(s[None])[1][0]))
            s2, r, done = env.step(a)
            memory.append((s, a, r, s2, float(done)))
            learn(net, [memory[-1]])                  # short-term memory
            s = s2
        if len(memory) >= BATCH:                       # replay after each game
            for _ in range(4):
                learn(net, rng.sample(memory, BATCH))
        scores.append(env.score)
        if env.score > best:
            best = env.score
        if (g + 1) % 25 == 0:
            print(f"game {g+1:4d}  score {env.score:2d}  best {best:2d}  "
                  f"avg25 {np.mean(scores[-25:]):.1f}  eps {eps:.2f}")
    return net, scores, ckpts, best


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    net, scores, ckpts, best = main(seed=seed)
    # final eval: greedy games
    rng = random.Random(1)
    env = Snake(rng)
    evals = []
    for _ in range(30):
        s, done = env.reset(), False
        while not done:
            s, _, done = env.step(int(np.argmax(net.forward(s[None])[1][0])))
        evals.append(env.score)
    print("greedy eval: mean %.1f max %d" % (np.mean(evals), max(evals)))
    ckpts.append({"game": len(scores), "best": best, "net": net.export()})
    json.dump({"scores": scores, "checkpoints": ckpts, "best": best,
               "eval_mean": float(np.mean(evals)), "eval_max": int(max(evals))},
              open("data.json", "w"), separators=(",", ":"))
