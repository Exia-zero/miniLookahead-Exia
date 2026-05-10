# emb_d4_t085 Summary

## Result

| Metric | Value |
| --- | ---: |
| Questions | 30 |
| QID range | 60-89 |
| Accuracy | 0.7333 |
| Total time | 22479.07 s |
| Total tokens | 536774 |
| Effective speed | 23.88 tokens/s |
| Median speed | 25.25 tokens/s |
| Weighted accept rate | 0.5098 |
| Median accept rate | 0.3713 |

`effective speed = total tokens / total time`，用于表示整组实验的总体吞吐。

## Comparison

| Metric | baseline_mw1 | emb_d1_t085 | emb_d4_t085 |
| --- | ---: | ---: | ---: |
| Accuracy | 0.7333 | 0.7000 | 0.7333 |
| Total time | 8812.56 s | 13601.15 s | 22479.07 s |
| Total tokens | 435515 | 478220 | 536774 |
| Effective speed | 49.42 tokens/s | 35.16 tokens/s | 23.88 tokens/s |

相对速度：

- `emb_d4` 比 `baseline_mw1` 慢约 `2.55x`；
- `emb_d4` 比 `emb_d1` 慢约 `1.65x`；
- `emb_d4` 的 accuracy 和 `baseline_mw1` 持平，但速度明显更慢。

## Key Observation

`depth=4` 没有带来速度收益。虽然 target 可以 batch 验证多个 prompt，但 draft 仍然需要串行生成多个 step。如果不能连续接受多个 draft step，后面的 draft/target 计算就会变成额外开销。

| Metric | emb_d1_t085 | emb_d4_t085 |
| --- | ---: | ---: |
| Total checks | 15907 | 16429 |
| Total time | 13601.15 s | 22479.07 s |
| Approx. sec/check | 0.855 | 1.368 |
| Weighted accept rate | 0.5566 | 0.5098 |
| Median accept rate | 0.3814 | 0.3713 |

d4 的 check 数量和 d1 接近，但总时间高很多，说明每个 decision 的平均成本更高。

## Long-Generation Effect

d4 有多道题接近 32k token：

| qid | time_s | tokens | acc_rate |
| ---: | ---: | ---: | ---: |
| 88 | 1619.58 | 32824 | 0.3393 |
| 61 | 1472.74 | 32816 | 0.3805 |
| 89 | 1349.15 | 32780 | 0.8997 |
| 77 | 1334.29 | 32814 | 0.6618 |
| 87 | 989.45 | 32795 | 0.7978 |

去掉 30k+ token 的长输出题后：

| Case | Weighted accept rate | Effective speed |
| --- | ---: | ---: |
| All 30 questions | 0.5098 | 23.88 tokens/s |
| Excluding 30k+ token questions | 0.3517 | 23.72 tokens/s |

这说明 d4 的 weighted accept rate 也被长输出题拉高。更典型的 accept rate 大约是 `0.35-0.37`。

## Main Takeaway

`emb_d4_t085` 的 accuracy 可以达到 `0.7333`，和 baseline 持平，但它没有带来加速。当前 accepted-step efficiency 不足以摊薄 4-step lookahead 的额外成本。

结论表述：

> `emb_d4_t085` preserves accuracy but is substantially slower. The current verifier/draft quality is not sufficient to amortize the overhead of 4-step draft generation and target batch verification, so depth=4 is not an effective speedup setting under the current configuration.
