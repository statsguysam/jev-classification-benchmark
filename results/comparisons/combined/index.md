# Exploratory paired comparisons

Differences are A minus B. All completed contrasts pass exact ordered test-content and matched-training checks, recomputed against frozen prepared files. Original manifest hashes remain visible and unchanged. The classical reference is the fixed TF-IDF MultinomialNB model, not the best observed classifier.

Intervals use 2,000 paired stratified test-item bootstrap resamples by default; each JSON records the actual count and bootstrap seed. These are multiple unadjusted exploratory contrasts, conditional on one selection seed and fixed runs. The contrast list was fixed for this report after some pilot outcomes were already visible; it is not a preregistration. Cross-family comparisons are descriptive and do not isolate model size, adaptation, or scoring protocol.

| Dataset / contrast | Accuracy delta [95% CI] | Macro F1 delta [95% CI] | Original manifest hashes |
|---|---|---|---|
| [sst2_qwen05_lora_minus_few4_seed42](sst2_qwen05_lora_minus_few4_seed42.json) | +0.0450 [+0.0000, +0.0900] | +0.0445 [-0.0007, +0.0900] | equal |
| [sst2_qwen4b_lora_minus_few4_seed42](sst2_qwen4b_lora_minus_few4_seed42.json) | -0.0200 [-0.0400, -0.0050] | -0.0200 [-0.0401, -0.0050] | equal |
| [sst2_qwen05_few4_minus_nb_k4_seed42](sst2_qwen05_few4_minus_nb_k4_seed42.json) | +0.3100 [+0.2250, +0.3951] | +0.3111 [+0.2259, +0.4000] | equal |
| [sst2_qwen4b_few4_minus_nb_k4_seed42](sst2_qwen4b_few4_minus_nb_k4_seed42.json) | +0.4150 [+0.3450, +0.4900] | +0.4162 [+0.3457, +0.4903] | equal |
| [sst2_luna_few4_minus_nb_k4_seed42](sst2_luna_few4_minus_nb_k4_seed42.json) | +0.4150 [+0.3400, +0.4900] | +0.4161 [+0.3412, +0.4916] | equal |
| [sst2_astra_few4_minus_nb_k4_seed42](sst2_astra_few4_minus_nb_k4_seed42.json) | +0.4350 [+0.3650, +0.5100] | +0.4362 [+0.3651, +0.5101] | equal |
| [sst2_jev_few4_minus_nb_k4_seed42](sst2_jev_few4_minus_nb_k4_seed42.json) | +0.4250 [+0.3500, +0.5000] | +0.4262 [+0.3512, +0.5051] | equal |
| [sst2_astra_few4_minus_luna_few4_seed42](sst2_astra_few4_minus_luna_few4_seed42.json) | +0.0200 [+0.0000, +0.0450] | +0.0200 [+0.0000, +0.0451] | equal |
| [sst2_qwen4b_few4_minus_astra_few4_seed42](sst2_qwen4b_few4_minus_astra_few4_seed42.json) | -0.0200 [-0.0550, +0.0100] | -0.0200 [-0.0550, +0.0100] | different; content audited |
| [sst2_jev_few4_minus_astra_few4_seed42](sst2_jev_few4_minus_astra_few4_seed42.json) | -0.0100 [-0.0300, +0.0100] | -0.0100 [-0.0300, +0.0100] | equal |
| [sst2_jev_few4_minus_qwen4b_few4_seed42](sst2_jev_few4_minus_qwen4b_few4_seed42.json) | +0.0100 [-0.0200, +0.0400] | +0.0100 [-0.0200, +0.0400] | different; content audited |
| [trec_qwen05_lora_minus_few4_seed42](trec_qwen05_lora_minus_few4_seed42.json) | -0.1350 [-0.1750, -0.1000] | -0.0435 [-0.0804, -0.0095] | equal |
| [trec_qwen4b_lora_minus_few4_seed42](trec_qwen4b_lora_minus_few4_seed42.json) | -0.0300 [-0.0800, +0.0150] | -0.1008 [-0.1793, -0.0317] | equal |
| [trec_qwen05_few4_minus_nb_k4_seed42](trec_qwen05_few4_minus_nb_k4_seed42.json) | -0.3250 [-0.4050, -0.2449] | -0.3976 [-0.4699, -0.3202] | equal |
| [trec_qwen4b_few4_minus_nb_k4_seed42](trec_qwen4b_few4_minus_nb_k4_seed42.json) | +0.3300 [+0.2550, +0.4000] | +0.3661 [+0.2967, +0.4360] | equal |
| [trec_luna_few4_minus_nb_k4_seed42](trec_luna_few4_minus_nb_k4_seed42.json) | +0.3500 [+0.2800, +0.4200] | +0.3622 [+0.2881, +0.4419] | equal |
| [trec_astra_few4_minus_nb_k4_seed42](trec_astra_few4_minus_nb_k4_seed42.json) | +0.4700 [+0.4000, +0.5400] | +0.4709 [+0.3982, +0.5474] | equal |
| [trec_jev_few4_minus_nb_k4_seed42](trec_jev_few4_minus_nb_k4_seed42.json) | +0.3550 [+0.2800, +0.4300] | +0.3884 [+0.3140, +0.4686] | equal |
| [trec_astra_few4_minus_luna_few4_seed42](trec_astra_few4_minus_luna_few4_seed42.json) | +0.1200 [+0.0800, +0.1600] | +0.1087 [+0.0703, +0.1480] | equal |
| [trec_qwen4b_few4_minus_astra_few4_seed42](trec_qwen4b_few4_minus_astra_few4_seed42.json) | -0.1400 [-0.1900, -0.0900] | -0.1048 [-0.1600, -0.0480] | different; content audited |
| [trec_jev_few4_minus_astra_few4_seed42](trec_jev_few4_minus_astra_few4_seed42.json) | -0.1150 [-0.1600, -0.0700] | -0.0826 [-0.1174, -0.0493] | equal |
| [trec_jev_few4_minus_qwen4b_few4_seed42](trec_jev_few4_minus_qwen4b_few4_seed42.json) | +0.0250 [-0.0300, +0.0750] | +0.0223 [-0.0335, +0.0750] | different; content audited |

Rebuild with `PYTHONPATH=src .venv/bin/python scripts/compare_combined_pilot.py`. Missing or incomplete runs are listed, never replaced with estimates. Existing comparison artifacts outside this index are not evidence of completion in the current rebuild.
