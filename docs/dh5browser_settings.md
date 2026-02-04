# dh5browser Settings Storage

## Overview

dh5browser uses ephyviewer's built-in settings persistence system, which stores settings in the Windows Registry (on Windows) or platform-appropriate locations on other operating systems.

## Settings Location

### Windows

Settings are stored in the Windows Registry at:

```
HKEY_CURRENT_USER\Software\ephyviewer\dh5browser_<filename>
```

Where `<filename>` is the stem of the DH5 file being viewed (without extension).

**Example:** For a file named `experiment_001.dh5`, settings are stored at:

```
HKEY_CURRENT_USER\Software\ephyviewer\dh5browser_experiment_001
```

### Linux/macOS

Settings are stored using Qt's QSettings, typically in:

- **Linux**: `~/.config/ephyviewer/dh5browser_<filename>.conf`
- **macOS**: `~/Library/Preferences/com.ephyviewer.dh5browser_<filename>.plist`

## What Settings Are Saved

### 1. Window Geometry and State (NEW)

As of the latest version, dh5browser now saves:

- **Window size and position** (`window_geometry`)
- **Dock widget visibility** (`window_state`) - This includes:
  - Which panels are open/closed (e.g., spike viewer, trace viewer, epoch viewer)
  - Panel positions (docked top/bottom/left/right, tabbed, or floating)
  - Panel sizes (width/height when docked)

This means if you close the spike viewer panel, it will remain closed when you reopen the same file.

### 2. Viewer Parameters

Each viewer type saves its parameters:

#### TraceViewer Parameters

- `ylim_max`, `ylim_min` - **Y-axis limits** (amplitude range)
  - **Important:** Once you set y-axis limits (manually or via auto-scale), they are preserved
  - Auto-scale is only applied on first load when no saved settings exist
  - Your custom y-axis limits persist across sessions and trial switches
- `xsize` - Time window width (seconds)
- `scale_mode` - Scaling mode: `'real_scale'`, `'same_for_all'`, or `'by_channel'`
- `auto_scale_factor` - Auto-scaling sensitivity (0-∞)
- `display_labels` - Show/hide channel labels
- Channel visibility (which channels are shown/hidden)
- Channel colors and display order

#### SpikeTrainViewer Parameters

- `xsize` - Time window width (seconds)
- Display mode and colors

#### EpochViewer Parameters

- `xsize` - Time window width (seconds)
- Channel colors
- Channel visibility

#### Navigation Toolbar

- Time position (intentionally **not** saved - always starts at trial beginning)
- Zoom level

### 3. Trial Selection

The trial index is intentionally **NOT** saved. Users expect to start at trial 0 (or the `--trial` argument) each time, not to resume where they left off in a previous session.

## Implementation Details

### DH5MainViewer Class

The custom `DH5MainViewer` class extends `ephyviewer.MainViewer` to add window state persistence:

```python
class DH5MainViewer(ephyviewer.MainViewer):
    """Extended MainViewer that persists window state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Note: Window state is NOT restored here because viewers haven't been added yet

    def restore_window_state(self):
        """Restore window geometry and dock widget state from settings.

        This is called AFTER all viewers have been added to ensure
        dock widgets exist before their state is restored.
        """
        self.restoreGeometry(self.settings.value("window_geometry"))
        self.restoreState(self.settings.value("window_state"))

    def save_all_settings(self):
        super().save_all_settings()  # Save viewer parameters
        # Additionally save window geometry and dock state
        self.settings.setValue("window_geometry", self.saveGeometry())
        self.settings.setValue("window_state", self.saveState())
```

**Important:** The `restore_window_state()` method is called at the end of `create_browser()`, after all viewers (trace, spike, epoch) have been added. This timing is critical - if we restore state before adding viewers, the dock widgets don't exist yet and the state restoration has no effect.

### Auto-Scale Behavior

To preserve user-configured y-axis limits, dh5browser intelligently controls when auto-scaling is applied:

1. **First time opening a file** (no saved settings): Auto-scale runs to set reasonable initial y-axis limits
2. **Reopening a file** (saved settings exist): Auto-scale is **skipped** to preserve your saved y-axis limits
3. **Switching between trials**: Y-axis limits remain constant (not auto-scaled per trial)

This means once you set your preferred y-axis range, it will be remembered and won't be overridden by auto-scale. If you want to reset to auto-scaled values, you can:

- Use the "Auto-scale" button in the viewer toolbar
- Clear the saved settings (see "Clearing/Resetting Settings" below)

### Settings Format

- Settings are stored as **pickled Python objects** (for viewer parameters) or **Qt binary data** (for window state)
- Each viewer is saved with a key like `viewer_<viewer_name>`
- Window geometry and state are saved as `window_geometry` and `window_state`

## Clearing/Resetting Settings

### Windows (Registry Editor)

1. Open Registry Editor (`regedit`)
2. Navigate to: `HKEY_CURRENT_USER\Software\ephyviewer`
3. Find the key named `dh5browser_<your_filename>`
4. Right-click and delete to reset all settings for that file

### Linux

```bash
rm ~/.config/ephyviewer/dh5browser_<filename>.conf
```

### macOS

```bash
defaults delete com.ephyviewer.dh5browser_<filename>
```

### Programmatic (Python)

```python
from PyQt5.QtCore import QSettings
settings = QSettings('ephyviewer', 'dh5browser_<filename>')
settings.clear()  # Clear all settings
```

## Per-File vs. Global Settings

**Important:** Settings are stored **per DH5 file** (based on the filename stem). This means:

- Each DH5 file has its own independent settings
- Window layout and y-axis limits for `experiment_001.dh5` won't affect `experiment_002.dh5`
- This allows you to customize the view for different types of experiments

## Troubleshooting

### Settings Not Being Saved

1. Check file permissions (ensure Qt can write to settings location)
2. Verify the application closes normally (settings are saved on `closeEvent`)
3. On Windows, ensure registry write permissions

### Settings Not Loading

1. Check if settings exist at the expected location
2. Verify filename matches (settings are per-file based on stem)
3. Check debug logs: `dh5browser --debug <file.dh5>` to see messages like:
   - "Restored window state from settings"
   - "Skipping auto-scale: using saved y-axis limits from settings"

### Y-axis Limits Keep Resetting

If your y-axis limits aren't being preserved:

1. Ensure you're closing the browser normally (not force-killing)
2. Settings are saved on window close - check that `closeEvent` is triggered
3. Clear settings once and reconfigure: `python test_panel_persistence.py yourfile.dh5 --clear`
4. Check debug logs to verify "Skipping auto-scale: using saved y-axis limits from settings"

### Corrupt Settings

If settings become corrupted (e.g., after ephyviewer upgrade), delete the settings as described in "Clearing/Resetting Settings" above.

## Version Compatibility

The ephyviewer parameter system includes compatibility checking. If the parameter tree structure changes between versions (e.g., after an update), settings will fail to load gracefully without crashing, and you'll see:

```
Not possible to restore settings
```

In this case, clear the settings for that file and reconfigure.
