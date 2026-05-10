# baseline_mw1 Summary

## Result

| Metric | Value |
| --- | ---: |
| Questions | 30 |
| QID range | 60-89 |
| Accuracy | 0.7333 |
| Total time | 8812.56 s |
| Total tokens | 435515 |
| Effective speed | 49.42 tokens/s |
| Median speed | 51.14 tokens/s |
| Avg time/question | 293.75 s |
| Median time/question | 232.16 s |

`effective speed = total tokens / total time`，用于表示整组实验的总体吞吐。

## Runtime Pattern

Baseline 没有 draft、target verification 或 accept/reject 逻辑，所以 accept rate 不适用。

最慢的题主要也是生成长度较长：

| qid | time_s | tokens | speed |
| ---: | ---: | ---: | ---: |
| 80 | 684.84 | 32000 | 46.73 |
| 87 | 647.63 | 30490 | 47.08 |
| 63 | 577.77 | 27768 | 48.06 |
| 88 | 560.69 | 26698 | 47.62 |
| 85 | 544.41 | 26052 | 47.85 |

生成长度和耗时高度相关：

| Correlation | Value |
| --- | ---: |
| tokens vs time | 0.9996 |

## Comparison Note

和 `emb_d1_t085_20260509_123146` 相比：

| Metric | baseline_mw1 | emb_d1_t085 |
| --- | ---: | ---: |
| Accuracy | 0.7333 | 0.7000 |
| Total time | 8812.56 s | 13601.15 s |
| Total tokens | 435515 | 478220 |
| Effective speed | 49.42 tokens/s | 35.16 tokens/s |

在当前 `max_workers=1` 的设置下，baseline 更快，并且 accuracy 略高。因此这组结果不支持 `emb_d1_t085` 相比 baseline 有端到端加速。

## Takeaway

`baseline_mw1` 是当前更公平的单题 latency baseline。后续比较 `emb` 方法时，应该继续使用相同题号范围、相同 `max_workers=1` 和相同 server 配置，否则 speedup 结论不可靠。
