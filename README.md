# NAP — Neuroimaging Agentic Processing

LLM-powered agent for automated EEG/MEG preprocessing with visual inspection.

## Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

## Usage

```bash
# With Claude API
export ANTHROPIC_API_KEY='your-key'
python -m nap.cli --project /path/to/output --data /path/to/raw.vhdr

# Without API key (simulated LLM responses)
python -m nap.cli --mock --project /path/to/output --data /path/to/raw.vhdr
```

Supported formats: `.vhdr`, `.edf`, `.bdf`, `.fif`, `.set`

Raw data is never modified. All outputs go to the project folder.
