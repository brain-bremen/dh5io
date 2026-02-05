# Usage

(Coming soon)

## Quick Start

Basic example of reading a DH5 file:

```python
from dh5io.dh5file import DH5File

with DH5File("example.dh5", "r") as dh5:
    # inspect file content
    print(dh5)
    
    # Get CONT group with id 1
    cont = dh5.get_cont_group_by_id(1)
    print(cont)
    
    # Get trial map
    trialmap = dh5.get_trialmap()
    print(trialmap)
```

For more detailed usage examples, see the [API Reference](api/index.md).
