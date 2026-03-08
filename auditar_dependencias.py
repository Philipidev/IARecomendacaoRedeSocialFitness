from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
NONE_LABEL = "- nenhum"
SKIP_DIRS = {
    ".cursor",
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "venv",
}
PACKAGE_ALIASES = {
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}
INDIRECT_PACKAGE_REASONS = {
    "tabulate": "uso indireto via pandas.DataFrame.to_markdown()",
}


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def local_top_level_modules(root: Path) -> set[str]:
    modules: set[str] = set()
    for path in iter_python_files(root):
        rel = path.relative_to(root)
        if len(rel.parts) == 1:
            modules.add(path.stem)
        else:
            modules.add(rel.parts[0])
    return modules


def normalize_package_name(module_name: str) -> str:
    return PACKAGE_ALIASES.get(module_name, module_name)


def parse_python_ast(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def iter_imported_modules(tree: ast.AST) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
            continue
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def is_external_top_level(module_name: str, local_modules: set[str]) -> bool:
    if module_name == "__future__":
        return False
    if module_name in sys.stdlib_module_names:
        return False
    if module_name in local_modules:
        return False
    return True


def requirements_declared(path: Path) -> set[str]:
    declared: set[str] = set()
    if not path.exists():
        return declared

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        base = re.split(r"[<>=!~\\[]", line, maxsplit=1)[0].strip()
        if base:
            declared.add(base)
    return declared


def external_imports_by_package(root: Path) -> dict[str, set[str]]:
    local_modules = local_top_level_modules(root)
    external: dict[str, set[str]] = defaultdict(set)

    for path in iter_python_files(root):
        tree = parse_python_ast(path)
        if tree is None:
            continue

        for module in iter_imported_modules(tree):
            top_level = module.split(".", 1)[0]
            if not is_external_top_level(top_level, local_modules):
                continue
            external[normalize_package_name(top_level)].add(
                path.relative_to(root).as_posix()
            )

    return dict(sorted(external.items()))


def uses_method_call(tree: ast.AST, method_name: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == method_name:
            return True
    return False


def indirect_dependencies(root: Path) -> dict[str, dict[str, object]]:
    detected: dict[str, dict[str, object]] = {}

    for path in iter_python_files(root):
        tree = parse_python_ast(path)
        if tree is None:
            continue

        if uses_method_call(tree, "to_markdown"):
            entry = detected.setdefault(
                "tabulate",
                {
                    "reason": INDIRECT_PACKAGE_REASONS["tabulate"],
                    "files": set(),
                },
            )
            files = entry["files"]
            assert isinstance(files, set)
            files.add(path.relative_to(root).as_posix())

    normalized: dict[str, dict[str, object]] = {}
    for package, payload in sorted(detected.items()):
        files = payload["files"]
        assert isinstance(files, set)
        normalized[package] = {
            "reason": payload["reason"],
            "files": sorted(files),
        }
    return normalized


def build_report(root: Path, requirements_path: Path) -> dict[str, object]:
    direct = external_imports_by_package(root)
    indirect = indirect_dependencies(root)
    declared = requirements_declared(requirements_path)
    used_packages = set(direct) | set(indirect)

    return {
        "requirements_path": requirements_path.relative_to(root).as_posix()
        if requirements_path.exists()
        else str(requirements_path),
        "declared_packages": sorted(declared),
        "direct_external_packages": {
            package: sorted(files) for package, files in sorted(direct.items())
        },
        "indirect_packages": indirect,
        "used_packages": sorted(used_packages),
        "missing_in_requirements": sorted(used_packages - declared),
        "declared_but_not_detected": sorted(declared - used_packages),
    }


def print_mapping_section(title: str, items: dict[str, object]) -> None:
    print(f"\n{title}")
    if not items:
        print(NONE_LABEL)
        return

    for package, payload in items.items():
        if isinstance(payload, list):
            print(f"- {package}: {', '.join(payload)}")
            continue
        if isinstance(payload, dict):
            reason = str(payload["reason"])
            files = payload["files"]
            assert isinstance(files, list)
            print(f"- {package}: {reason} ({', '.join(files)})")
            continue
        print(f"- {package}: {payload}")


def print_list_section(title: str, items: list[str]) -> None:
    print(f"\n{title}")
    if not items:
        print(NONE_LABEL)
        return

    for item in items:
        print(f"- {item}")


def print_report(report: dict[str, object]) -> None:
    print("=== Auditoria de Dependências Python ===")
    print(f"requirements.txt: {report['requirements_path']}")

    direct = report["direct_external_packages"]
    assert isinstance(direct, dict)
    print_mapping_section("Pacotes externos detectados por import direto:", direct)

    indirect = report["indirect_packages"]
    assert isinstance(indirect, dict)
    print_mapping_section("Pacotes detectados por uso indireto:", indirect)

    missing = report["missing_in_requirements"]
    assert isinstance(missing, list)
    print_list_section("Pacotes faltando no requirements.txt:", missing)

    unused = report["declared_but_not_detected"]
    assert isinstance(unused, list)
    print_list_section("Pacotes declarados mas não detectados:", unused)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audita dependências externas usadas no código Python do projeto."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Raiz do projeto a ser auditada.",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=ROOT / "requirements.txt",
        help="Caminho do requirements.txt a ser comparado.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime o relatório em JSON.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    requirements_path = (
        args.requirements.resolve()
        if args.requirements.is_absolute()
        else (root / args.requirements).resolve()
    )
    report = build_report(root, requirements_path)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print_report(report)


if __name__ == "__main__":
    main()
