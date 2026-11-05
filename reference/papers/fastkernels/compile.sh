#!/usr/bin/env bash
# Rebuild main.pdf (pdflatex + bibtex, repeated until references settle).
set -euo pipefail
cd "$(dirname "$0")"
exec latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
