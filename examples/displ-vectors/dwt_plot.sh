#!/bin/sh

: ${INPUT_DIR:=data/exp1}
: ${OUTPUT_FILE:=plot.dat}

echo "Creating plot ${OUTPUT_FILE} from ${INPUT_DIR}..."

rm -f "${OUTPUT_FILE}"

for f in "${INPUT_DIR}"/*; do
	C=$(basename "$f" .mat)
	NORM=$(./vectors "$f" /dev/null |& grep -E "[^_]norm = " | sed 's/.*= //')
	echo "for $C coefficients: $NORM"
	echo -e "$C\t$NORM" >> "${OUTPUT_FILE}"
done

mv "${OUTPUT_FILE}" "${OUTPUT_FILE}".unsorted
sort -n "${OUTPUT_FILE}".unsorted > "${OUTPUT_FILE}"

gnuplot plot.txt
inkscape -A error.pdf error.svg
