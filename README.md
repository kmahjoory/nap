# NAP — Neuroimaging Agentic Analysis Platform

LLM-powered agent for automated EEG/MEG preprocessing with visual inspection.

## Install

Requires Python 3.10+ and git.

```bash
mkdir <ProjectFolder>
cd <ProjectFolder>
python -m venv .venv  # Python 3.10+
source .venv/bin/activate
pip install git+https://github.com/kmahjoory/nap.git
# All dependencies are installed automatically
```

## Usage

```bash
# With Claude API
export ANTHROPIC_API_KEY='your-key'
nap --project ./output --data /path/to/eeg/

# Without API key (simulated LLM responses)
nap --mock --project ./output --data /path/to/eeg/

# Detailed output
nap --verbose --project ./output --data /path/to/eeg/

# Interactive mode (prompts for input)
nap --interactive --project ./output --data /path/to/eeg/
```

Supported formats: `.vhdr`, `.edf`, `.bdf`, `.fif`, `.set`

Raw data is never modified. All outputs go to the project folder.