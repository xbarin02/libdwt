#!/bin/sh

: ${INPUT_FILE:=data/coordinates.csv}
: ${OUTPUT_DIR:=data/exp1}
: ${PLOT_FILE:=plot.dat}

rm -f "${PLOT_FILE}"

: ${START:=1}
: ${STEP:=10}
: ${STOP:=200}

echo "Generating [${START}:${STEP}:${STOP}] test files from ${INPUT_FILE} into ${OUTPUT_DIR}..."

mkdir -p "${OUTPUT_DIR}"

for c in $(seq -w "${START}" "${STEP}" "${STOP}"); do
	OUTPUT_FILE=${OUTPUT_DIR}/$c.mat
	NORM=$(./vectors "${INPUT_FILE}" "${OUTPUT_FILE}" $c 0 |& grep "diff_norm = " | sed 's/.*= //')
	echo "for $c coefficients: $NORM"
	echo -e "$c\t$NORM" >> "${PLOT_FILE}"
done

mv "${PLOT_FILE}" "${PLOT_FILE}".unsorted
sort -n "${PLOT_FILE}".unsorted > "${PLOT_FILE}"

gnuplot plot.txt
inkscape -A error.pdf error.svg
