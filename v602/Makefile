all: build/v602.pdf

# hier Python-Skripte:
build/plotbragg.pdf build/plotemission.pdf build/detailspektrum.pdf build/Zn30.pdf build/Ga31.pdf build/Br35.pdf build/Sr38.pdf build/Zr40.pdf build/Zr40.pdf build/Rydberg.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/v602.pdf deklarieren:
build/v602.pdf: build/plotbragg.pdf build/plotemission.pdf build/detailspektrum.pdf build/Zn30.pdf build/Ga31.pdf build/Br35.pdf build/Sr38.pdf build/Zr40.pdf build/Zr40.pdf build/Rydberg.pdf
build/v602.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	v602.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
