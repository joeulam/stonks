PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate
NOTEBOOK := stonks-project-copy\ \(12\).ipynb
EXECUTED := artifacts/stonks-executed.ipynb

.PHONY: setup run test clean

setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE); pip install --upgrade pip
	$(ACTIVATE); pip install -r requirements.txt

run: $(VENV)/bin/activate
	$(ACTIVATE); mkdir -p artifacts
	$(ACTIVATE); papermill "$(NOTEBOOK)" "$(EXECUTED)"

test: $(VENV)/bin/activate
	$(ACTIVATE); pytest -q

clean:
	rm -rf $(VENV) artifacts
