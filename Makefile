.PHONY: install records sweep smoke finalize clean

install:
	pip install -r requirements.txt

# Build ticket records once (§1)
records:
	python run.py --stage records

# Full field-combo sweep (§2) — requires OPENAI_API_KEY + ANTHROPIC_API_KEY
sweep:
	python run.py --stage sweep

# End-to-end smoke test on 500 tickets, ZERO API spend (stub embeddings + LLM)
smoke:
	python run.py --dry-run --limit 500 --stage sweep

# Finalize a human-selected combo (runs stability first): make finalize COMBO=notes_weighted
finalize:
	python run.py --stage finalize --select $(COMBO)

clean:
	rm -rf results/combos results/*.json cache/*.npz cache/records*.json __pycache__
