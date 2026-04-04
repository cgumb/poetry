from __future__ import annotations

import importlib

MODULES = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "datasets",
    "huggingface_hub",
    "sentence_transformers",
    "mpi4py",
]

OPTIONAL_MODULES = [
    "streamlit",
]


def try_import(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        return True, str(version)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    failed = False
    print("Required modules:")
    for name in MODULES:
        ok, info = try_import(name)
        print(f"  {name}: {'OK' if ok else 'MISSING'} ({info})")
        failed = failed or (not ok)

    print("\nOptional modules:")
    for name in OPTIONAL_MODULES:
        ok, info = try_import(name)
        print(f"  {name}: {'OK' if ok else 'NOT INSTALLED'} ({info})")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
