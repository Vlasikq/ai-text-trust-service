"""Notebooks (04/06/07/08) ссылаются на app.features.stylometric и
app.preprocessing.text_cleaning. Тест: AST код-ячеек парсится и все
`from app.*` импорты резолвятся — без этого после миграции модулей
ноутбуки тихо ломаются, а исполняем мы их редко.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"

TARGET_NOTEBOOKS = [
    "04_stylometric.ipynb",
    "06_robustness.ipynb",
    "07_temporal_robustness.ipynb",
    "08_extended_analysis.ipynb",
]


def _extract_code(notebook_path: Path) -> str:
    with open(notebook_path, encoding="utf-8") as f:
        nb = json.load(f)
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            chunks.append("".join(source))
        else:
            chunks.append(source)
    return "\n\n".join(chunks)


def _app_imports(tree: ast.AST) -> list[str]:
    """Список модулей app.* из `from app.X import ...` и `import app.X`."""
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("app"):
                out.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("app"):
                    out.append(alias.name)
    return out


@pytest.mark.parametrize("name", TARGET_NOTEBOOKS)
def test_notebook_code_parses(name):
    path = NOTEBOOKS_DIR / name
    if not path.exists():
        pytest.skip(f"{name} отсутствует в репо")
    code = _extract_code(path)
    # Magic-команды (`%matplotlib inline`, `!pip install`) допустимы в notebook,
    # но не валидны для ast — отфильтровываем, чтобы тест ловил настоящие
    # синтаксические поломки, а не специфику Jupyter.
    cleaned = "\n".join(
        line for line in code.splitlines()
        if not line.lstrip().startswith(("%", "!", "?"))
    )
    ast.parse(cleaned)


@pytest.mark.parametrize("name", TARGET_NOTEBOOKS)
def test_notebook_app_imports_resolve(name):
    path = NOTEBOOKS_DIR / name
    if not path.exists():
        pytest.skip(f"{name} отсутствует в репо")
    code = _extract_code(path)
    cleaned = "\n".join(
        line for line in code.splitlines()
        if not line.lstrip().startswith(("%", "!", "?"))
    )
    tree = ast.parse(cleaned)
    modules = sorted(set(_app_imports(tree)))
    # Без app.* импортов проверять нечего — notebook самостоятельный.
    for mod in modules:
        importlib.import_module(mod)
