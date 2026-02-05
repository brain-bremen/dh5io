# Installation

This guide covers how to install **dh5io** and its command-line tools.

## Installing CLI Tools

The package includes three command-line tools:

- `dh5tree` - Display the structure of DH5 files
- `dh5merge` - Merge multiple DH5 files
- `dh5browser` - Interactive graphical browser for DH5 files

### Using uv tool (Recommended)

The [uv tool](https://docs.astral.sh/uv/guides/tools/) command allows you to install and run command-line tools in isolated environments without affecting your project dependencies:

```bash
# Install all CLI tools in an isolated environment
uv tool install dh5io

# For the browser tool, install with browser support
uv tool install dh5io[browser]
```

Once installed, you can run the tools directly:

```bash
# Display DH5 file structure
dh5tree mydata.dh5

# Merge DH5 files
dh5merge file1.dh5 file2.dh5 -o merged.dh5

# Launch interactive browser
dh5browser mydata.dh5
```

To verify the CLI tools are available:

```bash
dh5tree --help
dh5merge --help
dh5browser --help
```

See the [Optional Dependencies](#optional-dependencies) section above for additional features.

## Installing the Python package

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver. To install **dh5io** with uv:

```bash
uv pip install dh5io
```

or with pip:

```bash
pip install dh5io
```

### Optional Dependencies

The package includes several optional dependency groups for different features:

**Browser support** (interactive DH5 file viewer):

```bash
uv pip install dh5io[browser]
# or
pip install dh5io[browser]
```

**Neo integration** (for Neo object conversion):

```bash
uv pip install dh5io[neo]
# or
pip install dh5io[neo]
```

**All optional dependencies**:

```bash
uv pip install dh5io[all]
# or
pip install dh5io[all]
```

**Development dependencies**:

```bash
uv pip install dh5io[dev]
# or
pip install dh5io[dev]
```
