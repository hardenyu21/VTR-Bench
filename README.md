<div align="center">
  <h1>VTR-Bench</h1>
  <p><b>Video Text Rendering Benchmark and Agentic Generation</b></p>
  <p>
    <a href="https://huggingface.co/datasets/hardenyu/VTR-Bench"><b>Hugging Face Dataset</b></a>
    &nbsp; | &nbsp;
    <a href="#evaluation"><b>Evaluation</b></a>
    &nbsp; | &nbsp;
    <a href="#agentic-generation"><b>Agentic Generation</b></a>
  </p>
</div>

## Overview

Can a video generator render the requested text while preserving the scene,
layout, and motion described in a prompt? **VTR-Bench** evaluates these two
complementary aspects of video generation: **text fidelity** and **instruction
following**.

The benchmark places text within advertisements, scientific scenes, user
interfaces, cultural settings, and daily life. Each prompt specifies the exact
text to render and its carrier, together with the surrounding visual content.
Evaluation combines text transcription with a scene-specific checklist, rather
than treating correct spelling as a substitute for a faithful video.

This repository provides the benchmark annotations, a unified evaluation
pipeline, and an **Agentic I2V framework**. The framework plans a first frame,
inspects image and video candidates, and uses visual feedback to refine generation
while retaining the original video request.

## Benchmark

VTR-Bench contains **300 prompts**, **1,202 required-text blocks**, and **6,000
checklist questions**: 60 prompts per domain and 20 questions per prompt.

| Domain | ID prefix | Prompts |
| --- | --- | ---: |
| Advertisement | `AD` | 60 |
| Science | `SCI` | 60 |
| User Interface | `UI` | 60 |
| Culture | `CULT` | 60 |
| Daily Life | `LIFE` | 60 |

The final annotations are included in the repository:

- [prompts.json](vtr_bench/data/prompts.json): case IDs, original English prompts,
  scene metadata, and verbatim text references.
- [checklists.json](vtr_bench/data/checklists.json): per-case questions across
  Entity Presence, Spatial Relationship, Temporal Consistency, Motion Adherence,
  and Scene Attributes.

Generated videos are distributed separately on
[Hugging Face](https://huggingface.co/datasets/hardenyu/VTR-Bench/tree/main/generated_videos).
For evaluation, download one model's video folder or use your own generated
videos. Name each MP4 with its benchmark ID, for example `AD-0001.mp4`.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/hardenyu21/VTR-Bench.git
cd VTR-Bench
```

Both workflows use Linux, Python 3.12, NVIDIA GPUs, and `ffmpeg`/`ffprobe` on
`PATH`. Install evaluation and generation in **separate environments**: their
reference runtimes require different vLLM versions. Model checkpoints and API
credentials are supplied by the user.

## Evaluation

### 1. Set up the evaluator

Prepare a local Qwen3.8-27B or Qwen3.6-27B evaluator checkpoint, then install the
evaluation environment:

```bash
python3.12 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install -e . -r requirements/evaluation.txt
export VTR_EVALUATOR_MODEL=/path/to/evaluator-checkpoint
```

### 2. Provide a video path

```bash
vtr-bench evaluate /path/to/videos
```

A single benchmark video is also supported:

```bash
vtr-bench evaluate /path/to/AD-0001.mp4
```

The command automatically matches filenames to the bundled annotations and runs
both metrics sequentially. A directory should contain the MP4s directly, without
an additional nested model folder. Partial collections are supported: missing
IDs are reported but are not scored as failures. Unknown or duplicate IDs are
rejected, and compatible existing evaluations are resumed.

### 3. Read the scores

Results are written to `results/<input-name>/metrics.json`, with per-case outputs
and logs under `details/`. Use `--output /path/to/results` to choose a destination.

- **Checklist score ↑**: question-weighted yes rate, reported overall and across
  the five instruction-following dimensions.
- **Word Error Rate (WER) ↓**: transcription error against the required text,
  reported with `R+1`, `R+5`, and `R+10` hypothesis-token caps, where `R` is the
  reference token count. Case scores are bounded to `[0, 1]` and averaged equally
  across videos. The evaluator does not receive the target strings when
  transcribing the video.

The reference protocol uses BF16, tensor parallelism of 1, 2-FPS video sampling,
non-thinking mode, and at most three attempts per case. To check filenames without
loading the evaluator, use `vtr-bench evaluate /path/to/videos --dry-run`.

## Agentic Generation

The Central Agent uses `qwen3.7-plus` to plan and inspect candidates, Qwen-Image
(`qwen-image-3.0`) to generate or edit keyframes, and a resident MiniMax-H3 service
for image-to-video generation. It can revise a keyframe or motion guidance based
on visual feedback and select a final video from the generated candidates.

**Every video request includes the verbatim original prompt plus an additive
motion refinement.** A verbatim copy of the original prompt at the start of the
refinement is removed; the original request remains authoritative, including all
required text. Planning does not replace the original prompt.

### 1. Configure generation

Prepare the MiniMax-H3 FL2VA checkpoint at
`/path/to/models/MiniMax-H3/FL2VA`. The reference service uses four GPUs with
sufficient memory and disables CPU offload by default.

```bash
python3.12 -m venv .venv-gen
source .venv-gen/bin/activate
python -m pip install -e . -r requirements/generation-reference.txt
export VTR_BENCH_PROJECT_ROOT="$PWD"
export VTEXTBENCH_MODELS_ROOT=/path/to/models
export BAILIAN_API_KEY=YOUR_KEY
export BAILIAN_BASE_URL=https://YOUR_WORKSPACE.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Use an API endpoint with access to the configured chat and image models. See
[.env.example](.env.example) for the available environment variables; export them
explicitly in each shell, since the file is not loaded automatically.

### 2. Start the persistent video service

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vtr-bench agentic --serve
```

Keep this process running. Once loading finishes, the service is available at
`http://127.0.0.1:18123` and serializes requests through one shared H3 engine.
Keep it bound to loopback: the reference service has no network authentication.

### 3. Generate videos

In another shell, activate `.venv-gen`, enter the repository, and export the same
configuration as above. Run one case, or omit the ID to process all 300 prompts:

```bash
vtr-bench agentic --case-id AD-0001
vtr-bench agentic
```

The default output is **1344×768, 240 frames, 24 FPS, and 10 seconds**, with seed
42. H3 is requested to produce 241 frames; the conditioning frame is discarded,
and any audio is trimmed by the same 1/24-second offset.

Selected videos are saved as
`generated_videos/experiments/agentic/final_videos/<ID>.mp4`. Each case directory
retains its first and intermediate images, video candidates, inspection reports,
and agent trace. The default exploration budget is 20 actions, 30 API calls, and
three video generations. If exploration cannot finalize a result, a separate
rescue path may generate an additional candidate; its use is recorded in the
case outputs.

For non-agentic baseline generation, refer to the official
[vLLM-Omni recipes](https://github.com/vllm-project/vllm-omni/tree/main/recipes).

## Acknowledgements

The generation runtime builds on
[vLLM-Omni](https://github.com/vllm-project/vllm-omni), and keyframe generation
uses [Qwen-Image](https://github.com/QwenLM/Qwen-Image). We thank the developers of
these projects and the underlying models.

## License

The code is released under the [Apache-2.0 License](LICENSE). External models,
datasets, and dependencies are subject to their respective licenses and terms.
