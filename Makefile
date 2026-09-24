VENV    = qlbm-venv
PYTHON  ?= python3

.PHONY: clean check-python-version install ruff mypy test check-ci

clean:
	rm -rf $(VENV)
	find . -type f -name "*.pyc" -exec rm -f {} \;

check-python-version:
	@PYTHON_VERSION=$$($(PYTHON) --version 2>&1 | awk '{print $$2}'); \
	MAJOR_VERSION=$$(echo $$PYTHON_VERSION | cut -d. -f1); \
	MINOR_VERSION=$$(echo $$PYTHON_VERSION | cut -d. -f2); \
	if [ "$$MAJOR_VERSION" -ne 3 ] || [ "$$MINOR_VERSION" -lt 12 ] || [ "$$MINOR_VERSION" -gt 14 ]; then \
	    echo "Python version must be between 3.12 and 3.14"; \
	    exit 1; \
	fi

install: check-python-version pyproject.toml
	@ echo "Creating venv..."
	$(PYTHON) -m venv $(VENV)
	@ echo "Installing qlbm..."
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/pip install -e .[dev]
	@ echo "Installation successful!"

ruff:
	@ echo Running Ruff...
	$(VENV)/bin/ruff check qlbm
	@ echo Ruff was successful.

mypy:
	@ echo Running Mypy...
	$(VENV)/bin/mypy qlbm test --config-file pyproject.toml
	@ echo Mypy was successful

test:
	@ echo Running pytest...
	$(VENV)/bin/pytest test/unit --junitxml=pytest_report.xml
	@ echo All tests were successful.

doctest:
	@ echo Building docs...
	make -C docs doctest
	make -C docs html
	@ echo Docs were built successfully.

check-ci: ruff mypy test doctest
	@ echo "CI checks passed successfully."
