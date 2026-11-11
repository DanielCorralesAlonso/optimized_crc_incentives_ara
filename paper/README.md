# Paper

`main.tex` is the manuscript.  Its figures are read from `figures/`, which
`collect_figures.sh` fills from the pipeline's `outputs/`:

    bash run_paper.sh                 # from the repo root: every run the paper uses
    bash paper/collect_figures.sh     # copy the figures the paper includes
    cd paper && latexmk -pdf main.tex

The bibliography `sample.bib` (`\bibliography{sample}`) is not in the repository yet;
add it here before building.  LaTeX build files are ignored by git.
