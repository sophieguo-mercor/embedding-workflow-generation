.PHONY: install preprocess preprocess-dry preprocess-collect records sweep smoke finalize excel clean

install:
	pip install -r requirements.txt

# Clean the raw exports → data/cleaned/*.cleaned.xlsx (§1.1).
# Notes = deterministic regex; descriptions = Anthropic batch (needs ANTHROPIC_API_KEY).
preprocess:
	python run.py --stage clean

# Free: print the description-clean prompt + request/token estimate, no API spend.
preprocess-dry:
	python run.py --stage clean --dry-run

# Reconnect to an in-flight description batch and finish writing the twins.
preprocess-collect:
	python run.py --stage clean --collect

# Build ticket records once (§1). Reads the cleaned twins by default;
# add USE_RAW=1 to read the raw exports instead.
records:
	python run.py --stage records $(if $(USE_RAW),--use-raw,)

# Full field-combo sweep (§2) — requires OPENAI_API_KEY + ANTHROPIC_API_KEY
sweep:
	python run.py --stage sweep

# End-to-end smoke test on 500 tickets, ZERO API spend (stub embeddings + LLM)
smoke:
	python run.py --dry-run --limit 500 --stage sweep

# Finalize a human-selected combo (runs stability first): make finalize COMBO=notes_weighted
finalize:
	python run.py --stage finalize --select $(COMBO)

# Convert every combo JSON artifact into results/combos_xlsx/<name>.xlsx
excel:
	python to_excel.py

clean:
	rm -rf results/combos results/*.json cache/*.npz cache/records*.json __pycache__
