from aamutils.mapping import Mapper
from aamutils.algorithm.aaming import get_its, get_rc
from aamutils.utils import smiles_to_graph


def test_mapper_with_expected_rc():
    # Diels-Alder reaction example
    smiles = "C1=CC=CC1.C1=CC=CC1>>C1=CC2CC1C3C2C=CC3"
    mapper = Mapper()
    aam_smiles = mapper.get_aam(smiles, expected_rc=(6, 6))
    assert isinstance(aam_smiles, str)
    g, h = smiles_to_graph(aam_smiles)
    its = get_its(g, h)
    rc = get_rc(its)
    assert len(rc.nodes) == 6
    assert len(rc.edges) == 6


def test_mapper_infeasable():
    smiles = "C>>C"
    mapper = Mapper()
    aam_smiles = mapper.get_aam(smiles, expected_rc=(2, 2))
    assert aam_smiles is None
