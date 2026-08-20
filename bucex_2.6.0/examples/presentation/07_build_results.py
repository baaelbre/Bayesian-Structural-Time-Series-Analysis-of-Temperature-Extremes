"""Regenerate the complete set of presentation tables and figures."""
from settings import DIAGNOSTICS, workflow


if __name__ == "__main__":
    workflow().report(strict=True, diagnostics=DIAGNOSTICS)

