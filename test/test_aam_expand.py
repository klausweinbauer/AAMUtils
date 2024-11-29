import unittest
from aamutils.utils import smiles_to_graph
from aamutils.aam_expand import extend_aam_from_graph, extend_aam_from_rsmi


class TestExtendAAMFromGraph(unittest.TestCase):
    def test_extend_aam_from_graph(self):
        """
        Tests the extend_aam_from_graph function to ensure it correctly extends
        atom-atom mappings and returns the expected reaction SMILES.
        """
        rsmi = "CC[CH2:3][Cl:1].[N:2]>>CC[CH2:3][N:2].[Cl:1]"
        G, H = smiles_to_graph(rsmi)
        result_smiles = extend_aam_from_graph(G, H)
        expected = (
            "[Cl:1][CH2:3][CH2:5][CH3:4].[NH3:2]>>[ClH:1].[NH2:2][CH2:3][CH2:5][CH3:4]"
        )
        self.assertEqual(result_smiles, expected)

    def test_extend_aam_from_smiles(self):
        """
        Tests the extend_aam_from_graph function to ensure it correctly extends
        atom-atom mappings and returns the expected reaction SMILES.
        """
        rsmi = "CC[CH2:3][Cl:1].[N:2]>>CC[CH2:3][N:2].[Cl:1]"

        result_smiles = extend_aam_from_rsmi(rsmi)
        expected = (
            "[Cl:1][CH2:3][CH2:5][CH3:4].[NH3:2]>>[ClH:1].[NH2:2][CH2:3][CH2:5][CH3:4]"
        )
        self.assertEqual(result_smiles, expected)


# Run the unittest
if __name__ == "__main__":
    unittest.main()
