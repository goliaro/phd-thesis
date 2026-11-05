PDFLATEX = pdflatex -file-line-error
# PDFLATEX = pdflatex -file-line-error -halt-on-error
LATEX = latex

#default: pstopdf
default: pdf


bib: 
	$(PDFLATEX) paper
	bibtex paper || true
	$(PDFLATEX) paper && $(PDFLATEX) paper
dvi: 
	$(LATEX) paper

ps: dvi
	dvips -P pdf -o paper.ps -t letter paper

pstopdf: ps
	ps2pdf -dPDFSETTINGS=/printer -dEmbedAllFonts=true paper.ps paper.pdf

pdf: 
	$(PDFLATEX) paper

final: clean bib

clean: 
	rm -f *.aux paper.dvi paper.bbl paper.blg *.log *.out *.bbi paper.brf paper.ps paper.pdf 
