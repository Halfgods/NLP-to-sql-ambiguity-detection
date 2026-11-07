# Ambiguity classifier — self-evaluation (actual numbers)

Headline number: **unseen-schema test slice** (flights, streaming — never seen in training) accuracy **0.650**, macro-F1 **0.645**.

| slice | n | accuracy (thr) | macro-F1 (thr) | macro-F1 (raw argmax) |
|---|---|---|---|---|
| train | 525 | 0.730 | 0.727 | 0.732 |
| val | 120 | 0.733 | 0.730 | 0.729 |
| test | 105 | 0.676 | 0.668 | 0.648 |
| test_unseen | 240 | 0.650 | 0.645 | 0.648 |

**Statement: accuracy on the test slices is below 0.80** (slices: seen-schema test, unseen-schema test). Honest cause: the classifier is a linear head (LogisticRegression) on frozen all-MiniLM-L6-v2 embeddings of a single concatenated string; it must learn non-linear interactions (a ranking phrase must be read *against* which numeric columns exist in that very schema) that are only partially linear in embedding space, and the synthetic label semantics are strict (one rival column flips CLEAR to AMBIGUOUS).

### Confusion — test (seen schemas) (rows=gold, cols=pred; order CLEAR, AMBIGUOUS, UNDERSPECIFIED)

```
19  10  6
3  32  0
9  6  20
```

AMBIGUOUS P=0.667 R=0.914 F1=0.771; UNDERSPECIFIED P=0.769 R=0.571 F1=0.656

### Confusion — test_unseen (flights, streaming) (rows=gold, cols=pred; order CLEAR, AMBIGUOUS, UNDERSPECIFIED)

```
36  15  29
21  59  0
12  7  61
```

AMBIGUOUS P=0.728 R=0.738 F1=0.733; UNDERSPECIFIED P=0.678 R=0.762 F1=0.718

### Top confusions with example pairs

- [test] gold **CLEAR** -> predicted **AMBIGUOUS** (hospital-0003): "Pull up the most expensive patient visits." on schema `hospital-0003`
- [test] gold **CLEAR** -> predicted **AMBIGUOUS** (hospital-0011): "Show the most expensive encounters." on schema `hospital-0011`
- [test] gold **UNDERSPECIFIED** -> predicted **CLEAR** (hospital-0125): "Which visits are lengthy?" on schema `hospital-0125`
- [test] gold **UNDERSPECIFIED** -> predicted **CLEAR** (hospital-0127): "Pull up the long patient visits." on schema `hospital-0127`
- [test] gold **CLEAR** -> predicted **UNDERSPECIFIED** (hospital-0057): "List all the patients with their blood types." on schema `hospital-0057`
- [test] gold **CLEAR** -> predicted **UNDERSPECIFIED** (logistics-0071): "List all the warehouses with their phone numbers." on schema `logistics-0071`
- [test] gold **UNDERSPECIFIED** -> predicted **AMBIGUOUS** (hospital-0142): "Show the same one as last time." on schema `hospital-0142`
- [test] gold **UNDERSPECIFIED** -> predicted **AMBIGUOUS** (library-0117): "Pull up the recent checkouts." on schema `library-0117`
- [test] gold **AMBIGUOUS** -> predicted **CLEAR** (library-0018): "Show the most checked-out books." on schema `library-0018`
- [test] gold **AMBIGUOUS** -> predicted **CLEAR** (logistics-0102): "Pull up the latest shipments." on schema `logistics-0102`
- [test_unseen] gold **CLEAR** -> predicted **UNDERSPECIFIED** (flights-0037): "Show the flights with the longest kilometers flown." on schema `flights-0037`
- [test_unseen] gold **CLEAR** -> predicted **UNDERSPECIFIED** (flights-0039): "Get the passenger names of all the bookings." on schema `flights-0039`
- [test_unseen] gold **AMBIGUOUS** -> predicted **CLEAR** (flights-0004): "Pull up the longest flights." on schema `flights-0004`
- [test_unseen] gold **AMBIGUOUS** -> predicted **CLEAR** (flights-0006): "Show the longest flights." on schema `flights-0006`
- [test_unseen] gold **CLEAR** -> predicted **AMBIGUOUS** (flights-0068): "Retrieve the latest flights." on schema `flights-0068`
- [test_unseen] gold **CLEAR** -> predicted **AMBIGUOUS** (flights-0070): "Pull up the most recent flights." on schema `flights-0070`
- [test_unseen] gold **UNDERSPECIFIED** -> predicted **CLEAR** (flights-0036): "Show the flights with the longest miles flown." on schema `flights-0036`
- [test_unseen] gold **UNDERSPECIFIED** -> predicted **CLEAR** (flights-0038): "Show the flights with the longest kilometers flown." on schema `flights-0038`
- [test_unseen] gold **UNDERSPECIFIED** -> predicted **AMBIGUOUS** (streaming-0054): "Pull up the tracks with the highest play count." on schema `streaming-0054`
- [test_unseen] gold **UNDERSPECIFIED** -> predicted **AMBIGUOUS** (streaming-0056): "Pull up the tracks with the highest downloads total." on schema `streaming-0056`

### Regression guard

- test: predicted classes = ['AMBIGUOUS', 'CLEAR', 'UNDERSPECIFIED'] → no collapse ✔
- test_unseen: predicted classes = ['AMBIGUOUS', 'CLEAR', 'UNDERSPECIFIED'] → no collapse ✔

### Limitations

The dataset is synthetic: template-generated schemas from 7 families with randomized naming, so distribution shift from real SQLite files is expected. Labels follow our operational semantics (multiple schema-supported readings = AMBIGUOUS; missing criterion/reference or attribute the schema cannot provide = UNDERSPECIFIED; humans can reasonably disagree at mechanism boundaries (e.g. vague adjectives could be treated as ambiguous rather than underspecified. The embedding is capped at 256 tokens, so very long user schemas get truncated at the question side. Thresholds are additive biases tuned greedily on val — not a calibration.
