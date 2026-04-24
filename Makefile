.PHONY: build build-wheels bump-version check-deps-platforms check-dist check-universal-wheel clean dev format help install lint lock publish-pypi publish-testpypi test

UV := uv
DIST_FILES := dist/*
VERSION_BUMP ?= patch
DEPENDENCY_PYTHONS ?= 3.11 3.12 3.13 3.14
DEPENDENCY_PLATFORMS ?= x86_64-manylinux_2_28 aarch64-manylinux_2_28 aarch64-apple-darwin x86_64-pc-windows-msvc

.DEFAULT_GOAL := help

install:
	@$(UV) sync --locked

dev:
	@$(UV) sync --locked --dev

lock:
	@$(UV) lock --upgrade

format:
	@$(UV) run ruff format src scripts data tests

lint:
	@$(UV) run ruff format --check src scripts data tests
	@$(UV) run ruff check src scripts data tests

test:
	@$(UV) run pytest -q

build:
	@rm -rf build dist *.egg-info src/*.egg-info
	@$(UV) build

build-wheels: build check-universal-wheel

check-universal-wheel:
	@python -c 'from pathlib import Path; import sys, zipfile; wheels = sorted(Path("dist").glob("*.whl")); assert wheels, "No wheels found in dist/"; bad = []; [bad.append((str(wheel), meta)) for wheel in wheels for meta in [zipfile.ZipFile(wheel).read(next(name for name in zipfile.ZipFile(wheel).namelist() if name.endswith(".dist-info/WHEEL"))).decode()] if "Root-Is-Purelib: true" not in meta or "Tag: py3-none-any" not in meta]; assert not bad, "Non-universal wheel(s): " + ", ".join(name for name, _ in bad); print("Universal wheel(s) support Unix/Linux, macOS, and Windows:", ", ".join(str(wheel) for wheel in wheels))'

check-deps-platforms:
	@outdir="outputs/dependency-platform-check/$$(date -u '+%Y%m%dT%H%M%SZ')"; \
	mkdir -p "$$outdir"; \
	printf 'uv=%s\npython=%s\nplatforms=%s\npython_versions=%s\n' "$$($(UV) --version)" "$$(python --version)" "$(DEPENDENCY_PLATFORMS)" "$(DEPENDENCY_PYTHONS)" > "$$outdir/context.txt"; \
	exit_code=0; \
	for py in $(DEPENDENCY_PYTHONS); do \
		for platform in $(DEPENDENCY_PLATFORMS); do \
			name="py$${py}-$${platform}"; \
			echo "Checking $$name"; \
			if $(UV) pip compile pyproject.toml --python-version "$$py" --python-platform "$$platform" --only-binary :all: --no-sources --torch-backend cpu --no-header --no-annotate --output-file "$$outdir/$$name.requirements.txt" > "$$outdir/$$name.log" 2>&1; then \
				echo "PASS $$name" | tee -a "$$outdir/summary.txt"; \
			else \
				echo "FAIL $$name" | tee -a "$$outdir/summary.txt"; \
				tail -n 30 "$$outdir/$$name.log"; \
				exit_code=1; \
			fi; \
		done; \
	done; \
	echo "Dependency platform evidence: $$outdir"; \
	exit $$exit_code

bump-version:
	@if [ -n "$(VERSION)" ]; then \
		$(UV) version "$(VERSION)" --frozen; \
	else \
		$(UV) version --bump $(VERSION_BUMP) --frozen; \
	fi

check-dist: build
	@$(UV)x --from twine twine check $(DIST_FILES)

publish-testpypi: check-dist
	@pypi_token="$$(awk -F= '/^[[:space:]]*pypi_token[[:space:]]*=/ {sub(/^[^=]*=[[:space:]]*/, ""); print; exit}' .env)"; \
	if [ -z "$$pypi_token" ]; then echo "Missing pypi_token in .env"; exit 1; fi; \
	TWINE_USERNAME=__token__ TWINE_PASSWORD="$$pypi_token" $(UV)x --from twine twine upload --repository testpypi $(DIST_FILES)

publish-pypi:
	@$(MAKE) bump-version
	@$(MAKE) check-dist
	@pypi_token="$$(awk -F= '/^[[:space:]]*pypi_token[[:space:]]*=/ {sub(/^[^=]*=[[:space:]]*/, ""); print; exit}' .env)"; \
	if [ -z "$$pypi_token" ]; then echo "Missing pypi_token in .env"; exit 1; fi; \
	TWINE_USERNAME=__token__ TWINE_PASSWORD="$$pypi_token" $(UV)x --from twine twine upload --repository pypi $(DIST_FILES)

clean:
	@rm -rf .venv build dist *.egg-info .pytest_cache .coverage .mypy_cache .ruff_cache

help:
	@echo "Available targets:"
	@echo "  install  - Install locked runtime dependencies with uv"
	@echo "  dev      - Install locked runtime and dev dependencies with uv"
	@echo "  lock     - Refresh uv.lock"
	@echo "  format   - Format Python sources with ruff"
	@echo "  lint     - Check formatting and lint with ruff"
	@echo "  test     - Run tests with pytest"
	@echo "  build    - Build wheel and sdist"
	@echo "  build-wheels      - Build and verify universal wheel for Unix/Linux, macOS, and Windows"
	@echo "  bump-version      - Bump pyproject.toml version (VERSION_BUMP=patch or VERSION=1.2.3)"
	@echo "  check-deps-platforms - Verify dependency wheels resolve on supported OS/Python targets"
	@echo "  check-universal-wheel - Verify wheel tag is py3-none-any"
	@echo "  check-dist        - Build and validate wheel/sdist metadata with twine"
	@echo "  publish-testpypi  - Build, validate, and upload dist files to TestPyPI"
	@echo "  publish-pypi      - Bump version, build, validate, and upload dist files to PyPI"
	@echo "  clean    - Remove local build and cache artifacts"
