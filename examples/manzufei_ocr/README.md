# Manzufei OCR example

This example contains an optional configuration for a Chinese medical
admission-record extraction workflow.

It is not enabled by default. The default `config.yaml` keeps the original vLLM
flow and inline prompts unless users explicitly copy these settings or set the
matching environment variables.

## Enable the example prompts

Copy the `prompts` section from `config.example.yaml` into `config.yaml`:

```yaml
prompts:
  ocr_prompt_path: "./prompts/medical_admission_ocr_user_prompt.txt"
  extraction_system_prompt_path: "./prompts/medical_admission_extraction_system_prompt.txt"
  extraction_prompt_path: "./prompts/medical_admission_extraction_user_prompt_template.txt"
```

The Docker Compose file mounts `./prompts` at `/workspace/prompts`, so relative
paths work from the container working directory.

## Enable a custom backend

Set `custom_backend.enabled=true` and configure `custom_backend.base_url`, or set
`CUSTOM_BACKEND_URL` in the environment. Leave `custom_backend.enabled=false` to
use the original vLLM backend.
