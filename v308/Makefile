all: build/v308.pdf

# hier Python-Skripte:
build/plotHysterese.pdf build/plotHystereseFit.pdf build/plotHelmHoltz1.pdf build/plotHelmHoltz2.pdf build/plotHelmHoltz3.pdf build/plotLangeSpule.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/v308.pdf deklarieren:
build/v308.pdf: build/plotHysterese.pdf build/plotHystereseFit.pdf build/plotHelmHoltz1.pdf build/plotHelmHoltz2.pdf build/plotHelmHoltz3.pdf build/plotLangeSpule.pdf

build/v308.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	v308.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
