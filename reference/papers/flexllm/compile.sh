#!/bin/bash

# LaTeX compilation script with bibliography support
# This script runs the complete pdflatex -> bibtex -> pdflatex -> pdflatex sequence

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default filename
FILENAME="paper_arxiv25"

# Parse command line arguments
if [ $# -eq 1 ]; then
    FILENAME="${1%.tex}"  # Remove .tex extension if provided
fi

echo -e "${BLUE}Starting LaTeX compilation for ${FILENAME}.tex...${NC}"
echo ""

# Step 1: First pdflatex run
echo -e "${YELLOW}Step 1/4: Running pdflatex (first pass)...${NC}"
pdflatex -interaction=nonstopmode -file-line-error "${FILENAME}.tex"
if [ $? -eq 0 ] || [ -f "${FILENAME}.aux" ]; then
    echo -e "${GREEN}✓ First pdflatex pass completed${NC}"
else
    echo -e "${RED}✗ First pdflatex pass failed${NC}"
    exit 1
fi
echo ""

# Step 2: BibTeX run
echo -e "${YELLOW}Step 2/4: Running bibtex...${NC}"
bibtex "${FILENAME}"
if [ $? -eq 0 ] || [ -f "${FILENAME}.bbl" ]; then
    echo -e "${GREEN}✓ BibTeX completed${NC}"
else
    echo -e "${RED}✗ BibTeX failed${NC}"
    exit 1
fi
echo ""

# Step 3: Second pdflatex run
echo -e "${YELLOW}Step 3/4: Running pdflatex (second pass)...${NC}"
pdflatex -interaction=nonstopmode -file-line-error "${FILENAME}.tex"
echo -e "${GREEN}✓ Second pdflatex pass completed${NC}"
echo ""

# Step 4: Final pdflatex run
echo -e "${YELLOW}Step 4/4: Running pdflatex (final pass)...${NC}"
pdflatex -interaction=nonstopmode -file-line-error "${FILENAME}.tex"
echo -e "${GREEN}✓ Final pdflatex pass completed${NC}"
echo ""

# Check if PDF was generated
if [ -f "${FILENAME}.pdf" ]; then
    PDF_SIZE=$(wc -c < "${FILENAME}.pdf" | tr -d ' ')
    PAGES=$(pdfinfo "${FILENAME}.pdf" 2>/dev/null | grep "Pages:" | awk '{print $2}')
    if [ -z "$PAGES" ]; then
        PAGES="unknown"
    fi
    echo -e "${GREEN}🎉 Compilation successful!${NC}"
    echo -e "   PDF: ${FILENAME}.pdf"
    echo -e "   Size: ${PDF_SIZE} bytes"
    echo -e "   Pages: ${PAGES}"
    
    # Check if bibliography was included (rough heuristic)
    if [ "$PAGES" != "unknown" ] && [ "$PAGES" -gt 15 ]; then
        echo -e "${GREEN}✓ Bibliography appears to be included${NC}"
    fi
else
    echo -e "${RED}✗ PDF file was not generated${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Compilation complete! You can now open ${FILENAME}.pdf${NC}"