# Changelog for dh5io

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0-dev] - Unreleased

### Added

- **Wavelet support**: Added support for reading and writing WAVELET blocks with corresponding tests
- **dh5merge tool**: New command-line tool for merging multiple DH5 files
  - Support for merging WAVELET, TRIALMAP, and EV02 blocks
  - GUI selector for choosing files to merge
  - Auto-suggest output filename based on common prefix of input files
  - Proper INDEX handling and DATA concatenation for all supported block types
  - Preserve Operations from first file on merge
  - Handle differing calibrations when merging CONT
- **dh5browser tool**: New GUI tool for browsing and visualizing DH5 files
  - Scrolling through continuous data with segment annotations
  - Trial info widget and segment annotations
- **dh5neo implementation**: Added README and tests for dh5neo subpackage to load data from DH5 files using the NEO data model.
- **Specification documentation**: Added formal DH5 file format specification (revision 3.1)

## [0.2.1] - 2025-08-21

### Fixed

- Fixed wrong package name in version getter
- Fixed broken import in tests
- Fixed trialmap test

## [0.2.0] - 2025-08-21

Initial release on PyPI.
