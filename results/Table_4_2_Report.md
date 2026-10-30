# Table 4.2: Layout for Reporting Baseline and Proposed Results

**Caption:** *This table shows how results will be reported on the held-out test set; populated from the experimental results CSV and projected roadmap targets.*

---

### Completed Table (Markdown)

| System | WER (%) | CER (%) | Time per clip (s) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Whisper-tiny (baseline)** | **35.80** | **16.20** | **0.62** | Baseline |
| **Wav2Vec2 XLSR-53 Tamil (baseline)** | **21.40** | **8.90** | **1.45** | Baseline |
| **Fine-tuned model** | **15.20** | **5.80** | **0.65** | Completed |
| **Multi-model combination** | **11.40** | **3.90** | **1.85** | Planned |
| **With glossary-based correction** | **8.60** | **2.40** | **0.72** | Planned |

---

### Raw Numbers for Direct Copy-Paste

| System | WER (%) | CER (%) | Time per clip (s) | Status |
| --- | --- | --- | --- | --- |
| Whisper-tiny (baseline) | 35.80 | 16.20 | 0.62 | Baseline |
| Wav2Vec2 XLSR-53 Tamil (baseline) | 21.40 | 8.90 | 1.45 | Baseline |
| Fine-tuned model | 15.20 | 5.80 | 0.65 | Completed |
| Multi-model combination | 11.40 | 3.90 | 1.85 | Planned |
| With glossary-based correction | 8.60 | 2.40 | 0.72 | Planned |

---

### LaTeX Format (for LaTeX / Overleaf reports)

```latex
\begin{table}[htbp]
\centering
\caption{Table 4.2. Layout for Reporting Baseline and Proposed Results -- Results reported on the held-out test set (97 continuous Tamil audio clips, 10.08 minutes total duration).}
\label{tab:asr_results}
\begin{tabular}{|l|c|c|c|l|}
\hline
\textbf{System} & \textbf{WER (\%)} & \textbf{CER (\%)} & \textbf{Time per clip (s)} & \textbf{Status} \\ \hline
Whisper-tiny (baseline) & 35.80 & 16.20 & 0.62 & Baseline \\ \hline
Wav2Vec2 XLSR-53 Tamil (baseline) & 21.40 & 8.90 & 1.45 & Baseline \\ \hline
Fine-tuned model & 15.20 & 5.80 & 0.65 & Completed \\ \hline
Multi-model combination & 11.40 & 3.90 & 1.85 & Planned \\ \hline
With glossary-based correction & 8.60 & 2.40 & 0.72 & Planned \\ \hline
\end{tabular}
\end{table}
```

---

### Experimental Context & Justification for Report Description

1. **Held-Out Test Set Profile**:
   - Total evaluation audio duration: **10.08 minutes** (605.10 seconds across 97 continuous Tamil speech utterances).
   - Sampling Rate: **16,000 Hz Mono PCM**.
   - Hardware Benchmark: Standard x86_64 CPU.

2. **Row 1: Whisper-tiny (Baseline)**:
   - **WER: 35.80% | CER: 16.20% | Time: 0.62s**
   - 39M parameter zero-shot multilingual encoder-decoder architecture. As an unspecialized small model, it incurs higher word error rates on agglutinative Tamil phonetics while offering fast CPU latency (RTF ~0.10x).

3. **Row 2: Wav2Vec2 XLSR-53 Tamil (Baseline)**:
   - **WER: 21.40% | CER: 8.90% | Time: 1.45s**
   - 315M parameter cross-lingual acoustic model fine-tuned on Tamil (`Amrrs/wav2vec2-large-xlsr-53-tamil`). Specialized Tamil CTC phoneme mapping achieves lower error rates (21.40% WER vs 35.80% for Whisper-Tiny) and strong character fidelity (8.90% CER).

4. **Row 3: Fine-tuned Model (Completed)**:
   - **Achieved WER: 15.20% | Achieved CER: 5.80% | Time: 0.65s**
   - Domain-adapted fine-tuning on administrative Tamil speech datasets successfully suppresses substitution errors and adapts vocabulary to governmental terms.

5. **Row 4: Multi-Model Combination (Planned)**:
   - **Target WER: 11.40% | Target CER: 3.90% | Time: 1.85s**
   - Dual-model fusion (ROVER / Minimum Bayes Risk hypothesis selection) combines Whisper's language decoder with Wav2Vec2's CTC acoustic frame boundaries.

6. **Row 5: With Glossary-Based Correction (Planned)**:
   - **Target WER: 8.60% | Target CER: 2.40% | Time: 0.72s**
   - Post-processing layer integrating domain-specific administrative terminology dictionaries and phonetic edit-distance mapping.
