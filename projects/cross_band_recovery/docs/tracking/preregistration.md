# Cross-Band Satellite Recovery: Scientific Pre-Registration Protocol (`prereg-v1`)

**Project:** Cross-Band Satellite Imagery Recovery (Sentinel-1 SAR → Sentinel-2 Multi-Spectral Optical)  
**Date of Freeze:** 2026-10-18  
**Protocol Version:** `prereg-v1`  
**Status:** 🔒 **LOCKED & FROZEN PRIOR TO TEST SPLIT EXECUTION**  
**Assessor / Lead:** AI Agent (Antigravity) & Project Investigator  

---

## 1. Objective and Anti-Snooping Declaration

This pre-registration protocol establishes the complete evaluation criteria, primary/secondary metrics, decision rules, and cryptographic hashes of all model checkpoints and scripts **prior to any execution on the frozen holdout test set** (`data/tiles/test/`, 1,929 tiles). 

In accordance with best scientific practices (Roberts et al. 2017, Ploton et al. 2020), no evaluation on the test set is permitted until this document is finalized, committed, and cryptographically verified.

---

## 2. Model Selection (Derived Strictly from Validation Set — CHK-G7.2)

Model checkpoints to be evaluated were selected strictly using Validation loss minimization during Phase 6:

| Model Identifier | Seed | Best Val Epoch | Best Val Loss (L1 + SAM) | Best Val L1 | Role in Final Evaluation |
|---|:---:|:---:|:---:|:---:|---|
| **M1_S2only_s42** | 42 | 83 | 0.06306 | 0.0506 | Optical-only Baseline (Multi-Seed Ensemble) |
| **M1_S2only_s101** | 101 | 35 | 0.06503 | 0.0522 | Optical-only Baseline (Multi-Seed Ensemble) |
| **M1_S2only_s2024** | 2024 | 31 | 0.06661 | 0.0536 | Optical-only Baseline (Multi-Seed Ensemble) |
| **M2_Fusion_s42** | 42 | 31 | 0.06586 | 0.0530 | Fusion Model (Multi-Seed Ensemble) |
| **M2_Fusion_s101** | 101 | **108** | **0.06099** | **0.0489** | **Primary Benchmark Model** & xAI Analysis Target |
| **M2_Fusion_s2024** | 2024 | 65 | 0.06485 | 0.0525 | Fusion Model (Multi-Seed Ensemble) |

---

## 3. Declaration of Metrics (CHK-G7.4)

### 3.1 Primary Metric (Single Metric for RQ1 and RQ2)
- **Metric:** **Masked Spectral Mean Absolute Error ($\text{MAE}_{\text{masked}}$)**
- **Domain:** Surface Reflectance $[0.0, 1.0]$.
- **Definition:** Computed strictly over the masked (missing) optical channels for each tile, then averaged across test tiles:
  $$\text{MAE}_{\text{masked}} = \frac{1}{\sum_{c=1}^7 M_c} \sum_{c=1}^7 M_c \frac{1}{HW} \sum_{h=1}^H \sum_{w=1}^W |X_{\text{pred}}[c, h, w] - X_{\text{true}}[c, h, w]|$$
  where $M_c = 1$ if band $c$ is masked, and $0$ otherwise.

### 3.2 Co-Primary Spectral Signature Metric
- **Metric:** **Composite Spectral Angle Mapper ($\text{SAM}_{\text{composite}}$)** (degrees)
- **Definition:** Angle between the true 7-band spectrum and the composite reconstructed spectrum (where observed bands use true values and masked bands use predictions).

### 3.3 Secondary Metrics
1. **Root Mean Square Error (RMSE):** Penalizes large outlier deviations in reflectance.
2. **Structural Similarity Index (SSIM):** Measures spatial structural coherence (window size 7, dynamic range 1.0).
3. **Peak Signal-to-Noise Ratio (PSNR):** In decibels (dB), $20 \log_{10}(1.0 / \text{RMSE})$.
4. **Spectral Bias:** Mean signed error $(\hat{y} - y)$ on masked channels to assess systematic under/over-estimation.
5. **Recovery Rate:** Proportion of pixels satisfying absolute reconstruction error $\le \tau_{\text{band}}$.

---

## 4. Formalization of Provisional Thresholds (CHK-G7.1)

All criteria previously designated as `[PROVISIONAL]` are hereby permanently frozen with empirical grounding from Phase 3 (Baselines) and Phase 6 (Validation):

| Parameter | Frozen Value | Empirical Grounding / Justification |
|---|:---:|---|
| **Recovery Rate Tolerance ($\tau_{\text{band}}$)** | **$\le 0.050$ reflectance** | Derived from $1.5 \times \text{MAE}_{\text{B3}}$ on Val ($1.5 \times 0.0322 \approx 0.0483$). Matches Sentinel-2 surface reflectance atmospheric error tolerance ($\pm 0.05$). |
| **Graceful Degradation Margin** | **$\le 5.0\%$ ($\text{Ratio} \le 1.050$)** | Based on M2 Val degradation observed in G6 ($+1.20\%$). A margin of $\le 5\%$ establishes that withholding SAR does not cause catastrophic degradation. |
| **SHAP Additivity Tolerance** | **$\le 5.0\%$ relative error** | $\left\|\sum_i \phi_i - (f(x) - \mathbb{E}[f(x)])\right\| \le 0.05 \cdot |f(x) - \mathbb{E}[f(x)]|$. Confirms completeness axiom without numerical divergence. |
| **ROI Cloud Cover Upper Bound** | **$\le 5.0\%$** | Verified in Gate G1 (observed maximum ROI cloud cover was $1.25\%$, mean $0.18\%$). |

---

## 5. Statistical Decision Rules for RQ1 and RQ2 (Section 5.6)

For each of the 127 masking patterns, the incremental contribution of Sentinel-1 SAR is quantified by:
$$\Delta \text{MAE} = \text{MAE}(M2, S1_{\text{off}}) - \text{MAE}(M2, S1_{\text{on}})$$

1. **Positive Contribution of SAR ($\Delta \text{MAE} > 0$):**
   - Declared statistically supported if and only if the **95% Spatial Block Bootstrap Confidence Interval** (resampling $5 \times 5$ km spatial blocks, 1,000 iterations) **strictly excludes $0.0$** (lower bound $> 0$).
2. **Neutral / Unresolved ($\Delta \text{MAE} \approx 0$):**
   - If the 95% CI overlaps $0.0$, no definitive SAR advantage is claimed for that pattern.
3. **Negative Impact ($\Delta \text{MAE} < 0$):**
   - If the upper bound of the 95% CI is $< 0$, the condition is reported transparently as a case where SAR introduces noise or negative interference (no suppression of negative findings).
4. **Multiple Comparison Correction:**
   - When claiming overall family-wise significance across all 127 patterns, false discovery rate (FDR) control via Benjamini-Hochberg ($\alpha = 0.05$) will be applied.

---

## 6. Cryptographic Artifact Registry (CHK-G7.3)

Bitwise SHA-256 cryptographic hashes of all components participating in the final evaluation:

### 6.1 Trained Model Checkpoints
```
M1_S2only_s42:     ec190a32b70ef78d83d67a136bc6e116851201a6e51db64744397364ab198a68
M1_S2only_s101:    94d7fa17b4245daa488fab1f3004ad7ee7bdf9f8d2aceba7e08e4db63ef7f37f
M1_S2only_s2024:   05d65a0d5b554434cd7f11a60b4ac9addaa9a1afdcc4b3c3f170f6f624617b57
M2_Fusion_s42:     2b692f34db4bfe429af9cc3f234dcf916e7849427c23665ee78aa2af5307512b
M2_Fusion_s101:    d87ac80c62999cb4699366af2a2ddf4d30c623a0c3b09611e114efe2b2d67b45
M2_Fusion_s2024:   26b6d39fe910b8471cc60228ee09da3feace3fa5adcb2fcff2491016e1718231
```

### 6.2 Data Split and Config Manifests
```
spatial_split.parquet:            08aac02b765fa98c01d273f6fe34b757d60c9da8485788683aeeaa190eb8f2bb
val_baseline_benchmarks.parquet:  468fb58a4eba2f3855b8aa08899d1b4b0061551f76ffa1d3d3da07de68eb7534
configs/base.yaml:                28dacb030e297d56023ed048cc64aa9bea62afee9fe8c6a47773c19c2b07dc3c
```

### 6.3 Evaluation Scripts and Libraries
```
scripts/04_evaluate_final_test.py: a9ee74c9fbfe2e927f5dd1fd4f418b0a26a0c779e2fffd56e50f2311e84a2403
eval/metrics.py:                   cc1ddadbc1601e5aea6faee6474e5bf036c395df503049b2d703773b2086beba
models/vit_cross_band.py:          35f5327a15a5399b8ff08f83099d36fe68b569c5c78b93f819ba917345916348
models/channel_masking.py:         536f1b47c67e3dd0be7f54cf6812fdef8cce6ed8095f5d437fc636a76667a7d5
```

---

## 7. Approval and Freeze Confirmation

This pre-registration is confirmed and frozen. Any subsequent evaluation on the test set will verify these exact hashes before proceeding.
