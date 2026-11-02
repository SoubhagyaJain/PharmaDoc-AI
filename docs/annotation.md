# Manual annotation exercise

The reproducible generator supplies ground truth. Manual inspection checks that those labels actually align with the rendered/source text.

```powershell
uv run --locked pharmadoc-ai export-labels data/generated/manifest.json --output output/annotation_tasks.json
```

The export contains ten training-only prescription pages in Label Studio's task/annotation format. Configure a local Label Studio project using `configs/label_studio.xml`, then import the task JSON. Inspect patient/prescriber roles, repeated strengths, medication grouping, and dose/frequency spans. Never import test tasks during model selection.

Label Studio is optional and intentionally absent from the application environment. Install it in its own environment if you want its GUI. The export format is tested; launching and operating Label Studio is a separate learning exercise and is not claimed as an executed integration check.
