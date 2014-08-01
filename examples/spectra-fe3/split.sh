#!/bin/bash

echo "Splitting dataset into training and testing subsets..."

SDIR=train-test
DIR=../data

rm -rf -- "${SDIR}"

mkdir -- "${SDIR}"

cd -- "${SDIR}"

for f in "${DIR}"/fv_*.svm; do
	LINES_TOTAL=$(cat "${f}" | wc -l)
	LINES_TRAIN=$(echo "${LINES_TOTAL}/2" | bc)
	BASENAME=$(basename "${f/.svm/}")

	echo "Processing ${BASENAME}..."

	python $(which svm-subset) -s 1 "${f}" ${LINES_TRAIN} ${BASENAME}.train.svm ${BASENAME}.test.svm
done
