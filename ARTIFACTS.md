# Model Artifact Provenance

## `model_3_11.pt`

- Purpose: Jane Street puzzle model containing Python 3.11 cloudpickled bytecode
- Repository: `jane-street/2025-03-10`
- Source: <https://huggingface.co/jane-street/2025-03-10/blob/main/model_3_11.pt>
- Remote revision shown by Hugging Face: `2df04ba`
- Local size: `1,158,729,818` bytes
- Hugging Face SHA-256: `43aa7da7ccf749ae1fb95f8b7a6aa49536b73e27f0ac74cb90d5f824ccd484b2`
- Locally computed SHA-256: `43aa7da7ccf749ae1fb95f8b7a6aa49536b73e27f0ac74cb90d5f824ccd484b2`
- Verification result: exact match

The local digest was computed by streaming the complete file through SHA-256. The artifact was then
opened as a ZIP archive and its pickle stream was parsed with `pickletools`; no pickle payload was
executed.

Architecture recovery was subsequently performed with Python 3.11.14 inside the repository's
namespace and Landlock sandbox. Python 3.12 is not accepted for live analysis because the embedded
Python 3.11 bytecode is not forward-compatible at the opcode level.

The artifact itself is excluded from Git by the `model*.pt` ignore rule. Generated JSON inspection
reports are also excluded because they contain machine-local paths; reproduce them with:

```bash
python3 scripts/inspect_model.py --report outputs/reports/archive_metadata.json
```

## `model.pt`

- Purpose: Jane Street puzzle model for Python versions older than 3.11
- Local status: not downloaded
- Trusted SHA-256: not yet recorded
