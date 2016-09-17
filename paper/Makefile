LATEX       = pdflatex
CHECK_RERUN = grep "Rerun to get" $*.log

all: rv.pdf

%.pdf: %.tex
	${LATEX} $<
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "Done."
