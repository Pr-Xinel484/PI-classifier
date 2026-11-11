"""Data-free repository checks."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_python_sources_parse():
    for path in ROOT.rglob("*.py"):
        if ".venv" not in path.parts:
            ast.parse(path.read_text(),filename=str(path))

def test_ks_experiment_guarded():
    tree=ast.parse((ROOT/"kuramoto_sivashinsky.py").read_text())
    assert any(isinstance(node,ast.If) and "__name__" in ast.unparse(node.test) for node in tree.body)
    assert not any(isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) for node in tree.body)

def test_expected_files_present():
    for name in ["README.md",".gitignore","requirements.txt","persistence_image.py","birdsong_mel_classifier.py","birdsong_delay_classifier.py","kuramoto_sivashinsky.py"]:
        assert (ROOT/name).is_file()
