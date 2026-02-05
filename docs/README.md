# dh5io Documentation

This directory contains the Sphinx documentation for **dh5io**.

## Building the Documentation Locally

### Prerequisites

Install the documentation dependencies (optional with uv):

```bash
# Using uv - no need to install dependencies manually
# Just use: uv run make html

# Or install explicitly with uv
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### Build HTML Documentation

#### Using uv (Recommended)

```bash
cd docs
uv run make html
```

#### Using Make (without uv)

On Linux/macOS:

```bash
cd docs
make html
```

On Windows:

```bash
cd docs
make.bat html
```

The generated HTML documentation will be available in `build/html/index.html`.

### Other Build Formats

Build PDF documentation:

```bash
make latexpdf
```

Build EPUB documentation:

```bash
make epub
```

Clean build artifacts:

```bash
make clean
```

### Live Preview with Auto-Rebuild

For development, you can use `sphinx-autobuild` for live preview:

```bash
# Using uv (recommended)
cd docs
uv run --with sphinx-autobuild sphinx-autobuild source build/html

# Or install and run
uv pip install sphinx-autobuild
cd docs
sphinx-autobuild source build/html
```

Then open your browser to `http://127.0.0.1:8000`.

## Documentation Structure

- `source/` - Documentation source files (Markdown and reStructuredText)
  - `index.md` - Main documentation index
  - `installation.md` - Installation guide
  - `usage.md` - Usage examples
  - `cli_tools.md` - Command-line tools reference
  - `api/` - API reference documentation
  - `_static/` - Static files (images, CSS, etc.)
  - `_templates/` - Custom Sphinx templates
  - `conf.py` - Sphinx configuration
- `build/` - Generated documentation (gitignored)
- `requirements.txt` - Documentation build dependencies

## Read the Docs

This documentation is configured to build automatically on [Read the Docs](https://readthedocs.org/) when changes are pushed to the repository. The configuration is in `.readthedocs.yaml` at the repository root.

## Contributing

When adding new documentation:

1. Create new `.md` files in the `source/` directory
2. Add them to the appropriate `toctree` in `index.md` or other parent files
3. Use Markdown format (MyST parser is enabled)
4. Build locally to verify formatting and links
5. Commit both source files and any new static assets

### Markdown Features

The documentation uses [MyST Parser](https://myst-parser.readthedocs.io/) which supports:

- Standard Markdown syntax
- Sphinx directives using ` ```{directive}` syntax
- Cross-references using `{ref}` and `{doc}` roles
- Admonitions: `{note}`, `{warning}`, `{tip}`, etc.
- And many more advanced features

### Code Block Example

```python
from dh5io.dh5file import DH5File

with DH5File("example.dh5", "r") as dh5:
    print(dh5)
```

### Admonition Example

```markdown
:::{note}
This is a note admonition.
:::
```

Or using the shorter syntax:

````markdown
```{note}
This is a note admonition.
```
````

```

## License

The documentation is licensed under the same MIT License as the dh5io package.
```
