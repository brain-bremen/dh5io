# Changelog for dh5io

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **Wavelet support**: Added support for reading and writing WAVELET blocks with corresponding tests
- **dh5merge tool**: New command-line tool for merging multiple DH5 files
  - Support for merging WAVELET, TRIALMAP, and EV02 blocks
  - GUI selector for choosing files to merge
  - Auto-suggest output filename based on common prefix of input files
  - Proper INDEX handling and DATA concatenation for all supported block types
- **Specification documentation**: Added formal DH5 file format specification (revision 3.1)
  - Clarified WAVELET DATA shape and indexing ([N,M,F] layout: channels, time, frequencies)
  - Set INDEX.offset to 0-based indexing (was 1-based)

### Changed

- Pinned Python version to 3.11

### Fixed

- Fixed API change errors related to `DH5File.file` renamed to `DH5File._file`
- Fixed test issues

## [0.2.1] - 2025-08-21

### Fixed

- Fixed wrong package name in version getter
- Fixed broken import in tests
- Fixed trialmap test

## [0.2.0] - 2025-08-21

Initial release on PyPI.
