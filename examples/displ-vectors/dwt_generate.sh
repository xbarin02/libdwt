#!/bin/sh

: ${INPUT_FILE:=data/coordinates.csv}
: ${OUTPUT_DIR:=data/exp1}
: ${PLOT_FILE:=plot.dat}

rm -f "${PLOT_FILE}"

: ${START:=1}
: ${STEP:=1}
: ${STOP:=88}

: ${WAVELET:=2}

echo "Generating [${START}:${STEP}:${STOP}] test files from ${INPUT_FILE} into ${OUTPUT_DIR}..."

mkdir -p "${OUTPUT_DIR}"

for c in $(seq -w "${START}" "${STEP}" "${STOP}"); do
	OUTPUT_FILE=${OUTPUT_DIR}/$c.mat
	NORM=$(./vectors "${INPUT_FILE}" "${OUTPUT_FILE}" $c 0 ${WAVELET} |& grep "diff_norm = " | sed 's/.*= //')
	echo "for $c coefficients: $NORM"
	echo -e "$c\t$NORM" >> "${PLOT_FILE}"
done

mv "${PLOT_FILE}" "${PLOT_FILE}".unsorted
sort -n "${PLOT_FILE}".unsorted > "${PLOT_FILE}"

eval cp \"\${PLOT_FILE}\" \"\${PLOT_FILE/.dat/-${WAVELET}}.dat\"

gnuplot plot.txt
inkscape -A error.pdf error.svg
inkscape -A error-wavelets.pdf error-wavelets.svg
