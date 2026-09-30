# VTR-Bench

Video text rendering benchmark with two entrypoints: **evaluation** and
**Agentic I2V**. The original 300 prompts, 6,000 checklist questions, and
1,202 text references are included.

[Benchmark videos on Hugging Face](https://huggingface.co/datasets/hardenyu/VTR-Bench)
are downloaded separately. Videos must be named with their benchmark IDs,
such as `AD-0001.mp4` or `SCI-0060.mp4`.

## Evaluation

Use Python 3.12, a compatible NVIDIA GPU, a local Qwen3.8-27B or Qwen3.6-27B
evaluator checkpoint, and `ffmpeg`/`ffprobe`.

```bash
python3.12 -m venv .venv-eval
.venv-eval/bin/python -m pip install -e . -r requirements/evaluation.txt
source .venv-eval/bin/activate
export VTR_EVALUATOR_MODEL=/path/to/evaluator-checkpoint
```

Configure the evaluator once. Then give only the video path:

```bash
vtr-bench evaluate /path/to/videos
# Or evaluate a single benchmark video:
vtr-bench evaluate /path/to/AD-0001.mp4
```

The command runs Checklist and WER sequentially on the same evaluator and
automatically uses the bundled benchmark data. No task, prompt, checklist,
or profile selection is needed. It evaluates the videos present in the
folder; absent benchmark IDs are reported, not scored as failures. Unknown
or duplicate IDs are rejected. Existing matching evaluations resume.

Read `results/<input-name>/metrics.json` for both metrics. Detailed per-case
records and logs are retained under `details/` in that result directory.
Use `--output /path/to/results` to choose another destination, or `--dry-run`
to validate video names without loading the evaluator.

Checklist reports question-weighted yes rates and five dimension scores.
WER uses contextual tokenization and bounded R+1/R+5/R+10 transcription,
averaged equally across videos. Ground-truth strings are masked from WER
evaluator input. The supplied evaluation engines and inference/retry protocol
remain unchanged: TP=1, BF16, 2-FPS video sampling, non-thinking mode, and
at most three attempts per case. Core source hashes are recorded in
`vtr_bench/data/provenance.json`.

## Agentic I2V

Use a separate generation environment; the reference H3 backend and evaluation
runtime pin different vLLM versions.

```bash
python3.12 -m venv .venv-gen
.venv-gen/bin/python -m pip install -e . -r requirements/generation-reference.txt
source .venv-gen/bin/activate
export VTR_BENCH_PROJECT_ROOT="$PWD"
export VTEXTBENCH_MODELS_ROOT=/path/to/models
export BAILIAN_API_KEY=YOUR_KEY
export BAILIAN_BASE_URL=https://YOUR_WORKSPACE.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Prepare `MiniMax-H3/FL2VA` below the model directory, then start the resident
service once:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vtr-bench agentic --serve
```

In another configured shell, run one case or all 300:

```bash
vtr-bench agentic --case-id AD-0001
vtr-bench agentic
```

The Central Agent uses `qwen3.7-plus` and Qwen-Image to plan, generate, inspect,
and select candidates. Each video request preserves the verbatim original
prompt and appends a deduplicated motion refinement. The H3 service stays
loaded and serializes requests. It delivers 240 frames at 24 FPS (10 seconds),
discarding the conditioning frame and trimming audio by the same offset.

Outputs, first/intermediate images, and traces remain under
`generated_videos/experiments/agentic/`; selected videos are in `final_videos/`.
Defaults are 20 actions, 30 API calls, and three exploratory video generations.
Rescue is recorded separately and may generate outside the exploration budget.
See `.env.example` for configuration. Bind the reference service to loopback;
it has no network authentication.

For ordinary baseline generation, follow the official
[vLLM-Omni recipes](https://github.com/vllm-project/vllm-omni/tree/main/recipes).
No separate baseline, download, upload, or ablation CLI is bundled.

## License

Apache-2.0. External models and dependencies retain their own licenses.
