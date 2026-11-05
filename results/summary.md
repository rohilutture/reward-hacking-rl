# Results

Held-out MBPP test problems (300), greedy decoding. Mean over seeds with 95% t-interval (n = number of seeds).

| experiment | seeds | visible-test pass | all-tests pass | visible-but-not-hidden | hack-flag rate |
|---|---|---|---|---|---|
| base model | - | 54.3 | 45.7 | 8.7 | 0.7 |
| A | 3 | 62.2 ± 10.0 | 53.9 ± 10.5 | 8.3 ± 1.7 | 0.7 ± 0.0 |
| B | 3 | 62.1 ± 6.0 | 53.7 ± 5.0 | 8.4 ± 2.7 | 0.7 ± 0.0 |
| C | 3 | 62.3 ± 7.2 | 53.8 ± 5.8 | 8.6 ± 2.5 | 0.7 ± 0.0 |
| hardA | 3 | 62.1 ± 7.5 | 52.4 ± 5.4 | 9.7 ± 2.2 | 0.7 ± 0.0 |
| hardC | 3 | 60.9 ± 8.9 | 51.9 ± 5.5 | 9.0 ± 3.8 | 0.7 ± 0.0 |

## Training-time signals (mean over the last 10 training steps)

| experiment | seeds | training reward | visible pass | all-tests pass | reward minus all-pass (gap) | hack-flag rate |
|---|---|---|---|---|---|---|
| A | 3 | 61.1 ± 15.1 | 61.1 ± 15.1 | 54.9 ± 16.9 | 6.2 ± 2.8 | 0.5 ± 1.2 |
| B | 3 | 56.7 ± 19.7 | 61.2 ± 17.7 | 56.7 ± 19.7 | 0.0 ± 0.0 | 0.5 ± 1.4 |
| C | 3 | 57.6 ± 19.4 | 62.2 ± 16.8 | 57.7 ± 19.6 | -0.1 ± 0.2 | 0.5 ± 1.4 |
| hardA | 3 | 27.8 ± 2.4 | 27.8 ± 2.4 | 22.2 ± 6.7 | 5.6 ± 4.7 | 1.1 ± 2.2 |
| hardC | 3 | 23.7 ± 8.6 | 27.3 ± 6.2 | 23.7 ± 8.6 | 0.0 ± 0.0 | 1.1 ± 2.7 |
