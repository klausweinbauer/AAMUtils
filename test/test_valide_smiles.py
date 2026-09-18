from typing import List
import unittest
from aamutils.aam_expand import extend_aam_from_rsmi
from rdkit import Chem

PARTIAL_REACTIONS: List[str] = [
    "Br[CH2:2][CH3:1].[nH:3]1[cH:4][cH:5][c:6]2[cH:7][cH:8][c:9]([F:10])[n:11][c:12]1"
    "2>>[CH3:1][CH2:2][n:3]1[cH:4][cH:5][c:6]2[cH:7][cH:8][c:9]([F:10])[n:11][c:12]12"
    ".Br",
    "Cl[c:2]1[n:3][cH:4][c:5]2[nH:6][c:7]([CH:8]3[CH2:9][CH2:10][CH2:11]3)[cH:12][c:1"
    "3]2[cH:14]1.[NH3:1]>>[NH2:1][c:2]1[n:3][cH:4][c:5]2[nH:6][c:7]([CH:8]3[CH2:9][CH"
    "2:10][CH2:11]3)[cH:12][c:13]2[cH:14]1.Cl",
    "CC(C)(C)OC(=O)[NH:6][c:5]1[cH:4][n:3][c:2]([Cl:1])[cH:14][c:13]1[C:12]#[C:7][CH:"
    "8]1[CH2:9][CH2:10][CH2:11]1.O>>[Cl:1][c:2]1[n:3][cH:4][c:5]2[nH:6][c:7]([CH:8]3["
    "CH2:9][CH2:10][CH2:11]3)[cH:12][c:13]2[cH:14]1.CC(C)(C)OC(=O)O",
    "Br[CH2:9][CH2:8][C:6]([C:4]([O:3][CH2:2][CH3:1])=[O:5])([CH3:7])[S:19]([CH3:20])"
    "(=[O:21])=[O:22].[nH:10]1[cH:11][cH:12][c:13]([I:14])[c:15]([CH3:16])[c:17]1=[O:"
    "18]>>[CH3:1][CH2:2][O:3][C:4](=[O:5])[C:6]([CH3:7])([CH2:8][CH2:9][n:10]1[cH:11]"
    "[cH:12][c:13]([I:14])[c:15]([CH3:16])[c:17]1=[O:18])[S:19]([CH3:20])(=[O:21])=[O"
    ":22].Br",
    "F[c:8]1[c:2]([CH3:1])[c:3]([I:4])[cH:5][cH:6][n:7]1.[OH2:9]>>[CH3:1][c:2]1[c:3]("
    "[I:4])[cH:5][cH:6][nH:7][c:8]1=[O:9].F",
    "Br[c:9]1[c:8](-[c:7]2[cH:6][cH:5][cH:4][c:3]([O:2][CH3:1])[cH:30]2)[c:29]2[c:16]"
    "([s:15]1)[n:17][c:18]([CH3:19])[cH:20][c:21]2[NH:22][S:23](=[O:24])(=[O:25])[CH:"
    "26]1[CH2:27][CH2:28]1.OB(O)[c:10]1[cH:11][n:12][nH:13][cH:14]1>>[CH3:1][O:2][c:3"
    "]1[cH:4][cH:5][cH:6][c:7](-[c:8]2[c:9](-[c:10]3[cH:11][n:12][nH:13][cH:14]3)[s:1"
    "5][c:16]3[n:17][c:18]([CH3:19])[cH:20][c:21]([NH:22][S:23](=[O:24])(=[O:25])[CH:"
    "26]4[CH2:27][CH2:28]4)[c:29]23)[cH:30]1.OB(O)Br",
    "[CH3:1][CH2:2][O:3][C:4](=[O:5])[c:6]1[cH:7][nH:8][n:18][c:19]1[C:20]([F:21])([F"
    ":22])[F:23].Br[CH2:9][c:10]1[cH:11][cH:12][c:13]([O:14][CH3:15])[cH:16][cH:17]1>"
    ">[CH3:1][CH2:2][O:3][C:4](=[O:5])[c:6]1[cH:7][n:8]([CH2:9][c:10]2[cH:11][cH:12]["
    "c:13]([O:14][CH3:15])[cH:16][cH:17]2)[n:18][c:19]1[C:20]([F:21])([F:22])[F:23].B"
    "r",
    "CS(=O)(=O)O[CH2:25][CH2:24][CH:23]1[N:8]([C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:"
    "4])=[O:7])[CH2:9][CH2:10][N:11]([C:12](=[O:13])[O:14][CH2:15][c:16]2[cH:17][cH:1"
    "8][cH:19][cH:20][cH:21]2)[CH2:22]1.[nH:26]1[n:27][cH:28][c:29]2[cH:30][cH:31][cH"
    ":32][cH:33][c:34]12>>[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[N:8]1[CH2:"
    "9][CH2:10][N:11]([C:12](=[O:13])[O:14][CH2:15][c:16]2[cH:17][cH:18][cH:19][cH:20"
    "][cH:21]2)[CH2:22][CH:23]1[CH2:24][CH2:25][n:26]1[n:27][cH:28][c:29]2[cH:30][cH:"
    "31][cH:32][cH:33][c:34]12.CS(=O)(=O)O",
    "Br[CH2:20][c:21]1[cH:22][cH:23][c:24]([Br:25])[cH:26][cH:27]1.[CH3:1][CH2:2][N:3"
    "]([CH2:4][CH3:5])[CH2:6][CH2:7][CH2:8][CH2:9][O:10][c:11]1[cH:12][cH:13][c:14]2["
    "c:15]([cH:16]1)[cH:17][cH:18][nH:19]2>>[CH3:1][CH2:2][N:3]([CH2:4][CH3:5])[CH2:6"
    "][CH2:7][CH2:8][CH2:9][O:10][c:11]1[cH:12][cH:13][c:14]2[c:15]([cH:16]1)[cH:17]["
    "cH:18][n:19]2[CH2:20][c:21]1[cH:22][cH:23][c:24]([Br:25])[cH:26][cH:27]1.Br",
    "CC(C)(C)OC(=O)[n:15]1[c:14]2[cH:13][cH:12][c:11]([O:10][CH2:9][CH2:8][CH2:7][CH2"
    ":6][N:3]([CH2:2][CH3:1])[CH2:4][CH3:5])[cH:19][c:18]2[cH:17][cH:16]1.O>>[CH3:1]["
    "CH2:2][N:3]([CH2:4][CH3:5])[CH2:6][CH2:7][CH2:8][CH2:9][O:10][c:11]1[cH:12][cH:1"
    "3][c:14]2[nH:15][cH:16][cH:17][c:18]2[cH:19]1.CC(C)(C)OC(=O)O",
]


class TestExtendAAMValidation(unittest.TestCase):
    def is_valid_smiles(self, smiles: str) -> bool:
        if not isinstance(smiles, str) or not smiles.strip():
            return False
        return Chem.MolFromSmiles(smiles.strip()) is not None

    def assert_valid_reaction(self, reaction_smiles: str) -> None:
        self.assertIsInstance(reaction_smiles, str)
        self.assertIn(">", reaction_smiles)

        tokens = [
            token.strip()
            for side in reaction_smiles.split(">")
            for token in side.split(".")
            if token.strip()
        ]
        self.assertTrue(len(tokens) > 0)

        for token in tokens:
            self.assertTrue(
                self.is_valid_smiles(token),
                msg=f"Invalid molecule '{token}' in reaction '{reaction_smiles}'",
            )

    def test_extend_aam_from_partial_reactions(self) -> None:
        for idx, partial_rxn in enumerate(PARTIAL_REACTIONS):
            with self.subTest(index=idx, reaction=partial_rxn):
                self.assert_valid_reaction(partial_rxn)
                result_smiles = extend_aam_from_rsmi(partial_rxn)
                self.assert_valid_reaction(result_smiles)


if __name__ == "__main__":
    unittest.main()
