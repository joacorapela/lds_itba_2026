# all: ldsForNeuro.pdf ldsForNeuro_withNotes.pdf
all: ldsForNeuro.pdf
# all: ldsForNeuro_withNotes.pdf

ldsForNeuro.pdf: ldsForNeuro.tex ldsForNeuro_content.tex

ldsForNeuro_withNotes.pdf: ldsForNeuro_withNotes.tex ldsForNeuro_content.tex

%.pdf: %.tex
	pdflatex $<
	bibtex $*.aux
	pdflatex $<
	pdflatex $<
