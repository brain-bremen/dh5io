# Welcome to dh5io's documentation!

**dh5io** is a Python package for handling [DAQ-HDF5](https://github.com/cog-neurophys-lab/DAQ-HDF5) (`*.dh5`) files.
The DH5 format is a hierarchical data format based on [HDF5](https://www.hdfgroup.org/solutions/hdf5/)
designed for storing and sharing neurophysiology data, used in the Brain Research Institute
of the University of Bremen since 2005.

## Quick Links

- **[Installation Guide](installation.md)** - Get started with pip or uv
- **[CLI Tools](cli_tools.md)** - Command-line tools for working with DH5 files
- **[Usage Examples](usage.md)** - Learn how to use the library
- **[API Reference](api/index.md)** - Detailed API documentation

## Quick Start

Install [CLI tools (dh5merge, dh5tree, dh5browser)](cli_tools.md) using uv (recommended):

```bash
uv tool install dh5io
```

Install dh5io library for using in Python scripts:

```bash
uv pip install dh5io
```

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
```

```{toctree}
:maxdepth: 2
:caption: User Guide

usage
cli_tools
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: About

changelog
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
