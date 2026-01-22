# K-He SPS Example

Compute spin-exchange cross-sections for K-He collisions using the Siegert pseudostate method.

## Usage

```bash
python run_everything.py electron-structure-outputs.json
```

### Options

- `--sections`: Which SPS sections to run (default: `all-identities`)
- `--start-identity`: Zero-based start index for processing identities (default: 0)
- `--max-identities`: Maximum number of identities to process (default: all)

### Example

Process only the first 3 identities:

```bash
python run_everything.py electron-structure-outputs.json --max-identities 3
```

## Input Schema

The input JSON file must conform to the `ElectronStructureOutputs` schema defined in `schema.py`:

```python
class ElectronStructureOutputs(msgspec.Struct, frozen=True):
    R: list[float]                    # Radial grid points (Bohr)
    all_V: list[list[float]]          # Potentials [n_points, n_identities] (Hartree)
    all_identities: list[str]         # Identity labels (e.g., "4s", "4p_y")
    all_rho_alpha: list[list[float]]  # Alpha spin density [n_points, n_identities]
    all_rho_beta: list[list[float]]   # Beta spin density [n_points, n_identities]
```

To create your own input file, either:
1. Build the JSON manually with the required fields
2. Use the schema directly:

```python
from schema import ElectronStructureOutputs
import msgspec.json

data = ElectronStructureOutputs(
    R=[2.5, 3.0, ...],
    all_V=[[...], ...],
    all_identities=["4s", "4p_y", ...],
    all_rho_alpha=[[...], ...],
    all_rho_beta=[[...], ...],
)
with open("my-outputs.json", "wb") as f:
    f.write(msgspec.json.encode(data))
```
