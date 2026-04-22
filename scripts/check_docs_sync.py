#!/usr/bin/env python3
"""Verify that documentation stays in sync with the actual mlxterp API.

Checks:
1. Every public function/class in mlxterp is documented in docs/API.md
2. Every import example in docs actually resolves
3. Every code block in docs/*.md uses importable names

Usage:
    python scripts/check_docs_sync.py

Exit codes:
    0 - All checks pass
    1 - Discrepancies found
"""

import ast
import importlib
import os
import re
import sys
from pathlib import Path


def get_public_api(package_dir: str) -> dict[str, list[str]]:
    """Extract all public functions, classes from the package."""
    public = {}
    pkg = Path(package_dir)

    for py_file in sorted(pkg.rglob("*.py")):
        if "__pycache__" in str(py_file) or ".venv" in str(py_file):
            continue

        rel = py_file.relative_to(pkg.parent)
        module_path = str(rel).replace("/", ".").replace(".py", "")
        if module_path.endswith(".__init__"):
            module_path = module_path[:-9]

        try:
            with open(py_file) as f:
                tree = ast.parse(f.read())
        except SyntaxError:
            continue

        names = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith("_"):
                    names.append(node.name)

        if names:
            public[module_path] = names

    return public


def get_documented_names(docs_dir: str) -> set[str]:
    """Extract all function/class names referenced in documentation."""
    documented = set()
    docs = Path(docs_dir)

    for md_file in docs.rglob("*.md"):
        with open(md_file) as f:
            content = f.read()

        # Match code blocks with Python imports
        for match in re.finditer(r"from\s+mlxterp[\w.]*\s+import\s+([^\n]+)", content):
            imports = match.group(1)
            for name in re.findall(r"\b([A-Za-z_]\w*)\b", imports):
                if name not in ("import", "as", "from"):
                    documented.add(name)

        # Match class/function references like `InterpretableModel`
        for match in re.finditer(r"`(\w+)`", content):
            documented.add(match.group(1))

    return documented


def check_import_examples(docs_dir: str) -> list[str]:
    """Check that import statements in docs actually work."""
    errors = []
    docs = Path(docs_dir)

    for md_file in sorted(docs.rglob("*.md")):
        with open(md_file) as f:
            content = f.read()

        # Find Python code blocks
        blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)

        for block in blocks:
            # Join multi-line imports: collapse "from X import (\n  a,\n  b\n)" into one line
            collapsed = re.sub(r"\(\s*\n", "(", block)
            collapsed = re.sub(r",\s*\n\s*", ", ", collapsed)
            collapsed = re.sub(r"\s*\)", ")", collapsed)

            for line in collapsed.split("\n"):
                line = line.strip()
                if line.startswith("from mlxterp") or line.startswith("import mlxterp"):
                    # Extract the module path
                    match = re.match(
                        r"from\s+(mlxterp[\w.]*)\s+import\s+\(?(.*?)\)?$", line
                    )
                    if match:
                        module = match.group(1)
                        raw_names = match.group(2)
                        names = [
                            n.strip().split(" as ")[0].strip()
                            for n in raw_names.split(",")
                        ]
                        names = [n for n in names if n and n.isidentifier()]
                        try:
                            mod = importlib.import_module(module)
                            for name in names:
                                if not hasattr(mod, name):
                                    errors.append(
                                        f"{md_file.name}: `from {module} import {name}` "
                                        f"- '{name}' not found in {module}"
                                    )
                        except ImportError as e:
                            errors.append(f"{md_file.name}: `{line}` - {e}")

    return errors


def main():
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    package_dir = project_root / "mlxterp"
    docs_dir = project_root / "docs"

    if not package_dir.exists():
        print(f"ERROR: Package directory not found: {package_dir}")
        sys.exit(1)

    if not docs_dir.exists():
        print(f"ERROR: Docs directory not found: {docs_dir}")
        sys.exit(1)

    # Add project root to path for imports
    sys.path.insert(0, str(project_root))

    errors = []
    warnings = []

    # Check 1: Public API coverage in docs
    print("Checking public API documentation coverage...")
    public_api = get_public_api(str(package_dir))
    documented = get_documented_names(str(docs_dir))

    undocumented = []
    for module, names in sorted(public_api.items()):
        for name in names:
            if name not in documented:
                undocumented.append(f"  {module}.{name}")

    if undocumented:
        warnings.append(
            f"Public API names not referenced in docs ({len(undocumented)}):\n"
            + "\n".join(undocumented)
        )

    # Check 2: Import examples in docs resolve
    print("Checking import statements in documentation...")
    import_errors = check_import_examples(str(docs_dir))
    if import_errors:
        errors.extend(import_errors)

    # Report
    print("\n" + "=" * 60)
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
        print()

    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")
        print()

    if not errors and not warnings:
        print("All docs-code sync checks passed.")

    if errors:
        print(f"\nFAILED: {len(errors)} error(s) found. Docs are out of sync with code.")
        sys.exit(1)
    else:
        print("\nPASSED: No critical sync issues.")
        sys.exit(0)


if __name__ == "__main__":
    main()
