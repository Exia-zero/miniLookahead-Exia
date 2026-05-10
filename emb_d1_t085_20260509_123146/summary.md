# emb_d1_t085 Summary

## Result

| Metric | Value |
| --- | ---: |
| Questions | 30 |
| QID range | 60-89 |
| Accuracy | 0.70 |
| Total time | 13601.15 s |
| Total tokens | 478220 |
| Effective speed | 35.16 tokens/s |
| Median speed | 39.14 tokens/s |
| Weighted accept rate | 0.5566 |
| Median accept rate | 0.3814 |

`effective speed = total tokens / total time`，比直接平均每题 speed 更适合表示整组实验的总体速度。

## Key Observation

`weighted accept rate` 看起来较高，但主要被 qid 80 拉高。

| Case | Weighted accept rate | Effective speed |
| --- | ---: | ---: |
| All 30 questions | 0.5566 | 35.16 tokens/s |
| Excluding qid 80 | 0.3740 | 36.87 tokens/s |

qid 80 是明显异常点：

| qid | time_s | tokens | accepts | checks | acc_rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 80 | 1519.39 | 32770 | 4694 | 4783 | 0.9814 |

它接近 token 上限，并且几乎所有 draft step 都被接受，说明 draft 和 target 在局部 step 上高度一致，但整体生成没有及时结束。

## Main Takeaway

`emb_d1_t085` 的单步 draft verification 整体比旧的多步 lookahead 更稳定，accuracy 达到 `0.70`，有效速度为 `35.16 tokens/s`。

但 accept rate 不能单独解释速度。当前结果里，耗时主要由生成长度决定：

| Correlation | Value |
| --- | ---: |
| tokens vs time | 0.9552 |
| checks vs time | 0.7873 |
| accept rate vs speed | -0.5733 |

因此，高 accept rate 不一定代表更快；如果模型沿着一条很长的 draft/target 一致轨迹继续生成，反而会变慢。

## Recommended Reporting

建议报告中同时写：

- Accuracy: `0.70`
- Effective speed: `35.16 tokens/s`
- Median accept rate: `0.3814`
- Weighted accept rate: `0.5566`
- Weighted accept rate without qid 80: `0.3740`

结论表述：

> `emb_d1_t085` shows reasonable accuracy and stable speed, but the accepted-step rate is sensitive to long-generation outliers. The main runtime driver is generation length rather than accept rate alone.
