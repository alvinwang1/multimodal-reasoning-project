# Arjun's local Idea 4 run (Mac M2 Max, MPS, bf16): kept as a cross-hardware replication

The official Idea 4 pipeline and results are Nikkita's (`src/midterm/surrogates.py`,
`src/midterm/idea4_surrogates.py`, `results/midterm/idea4/`, NVIDIA L4). This folder keeps the
equivalent pipeline that was run on the laptop over the same 120 test items:

- `surrogates.py`, `idea4_surrogates.py`: the local generation and analysis scripts
- `data_arjun.py`: the local copy of `src/data.py` with the PyAV fallback for item 685 (same fix as on main)
- `results/midterm_arjun/generated_surrogates.json`: Whisper + Qwen surrogates (with the end-of-turn fix)
- `results/midterm_arjun/generated_surrogates_v1_no_eos.json`: the first, broken run, where generation did not
  stop at the end of the turn and Qwen invented "Human:" dialogue turns (kept as a failure example)
- `results/midterm_arjun/raw_surrogate.jsonl`: scores for the 5 generated-surrogate configs (git-ignored, large)
