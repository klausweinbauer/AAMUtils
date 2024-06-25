# Atom-atom-mapping Utils

A collection of atom-atom-mapping utility functions. 

## Installation

TODO

## Usage

The input is a list of partial atom-atom-maps (AAMs). Data is read line-by-line
from a text file. Each line should contain one reaction SMILES.

Here is a simple example extending the partial AAM to a complete AAM. First
generate the input data:

```bash
echo "CCC[Cl:1].[N:2]>>CCC[N:2].[Cl:1]" > testinput.txt
```

Next, run AAMUtils to expand the partial AAMs. 

```bash 
python3 ./aamutils/__main__.py expand testinput.txt
```

The output is written to 'testinput_extended.json'. 

```bash 
cat testinput_extended.json 
```

```json 
[
    {
        "input": "CCC[Cl:1].[N:2]>>CCC[N:2].[Cl:1]",
        "expanded_aam": "[Cl:1][CH2:5][CH2:4][CH3:3].[NH3:2]>>[ClH:1].[NH2:2][CH2:5][CH2:4][CH3:3]",
        "ilp_status": "Optimal",
        "optimization_result": 2.0,
        "invalid_reaction_center": false,
        "reaction_edges": 3
    }
]
```

## Functionality
Here is an overview of implemented functionallity:

- SMILES to graph and graph to SMILES parsing
- Reaction center validity checks
- ITS graph generation
- Expand partial AAM to complete AAM on balanced reactions
- AAM based on minimal chemical distance (MCD) for balanced reactions

## License

This project is licensed under MIT License - see the [License](LICENSE) file
for details.

## Acknowledgments

This project has received funding from the European Unions Horizon Europe
Doctoral Network programme under the Marie-Skłodowska-Curie grant agreement No
101072930 (TACsy -- Training Alliance for Computational)
