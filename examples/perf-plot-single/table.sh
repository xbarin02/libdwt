#!/bin/bash

DIR=data
SCALE=2147483647
UNIT=1000000000
BASE=21.9

for i in 3 4 11 12; do
	for t in 1 2 4; do
		LAST=$(tail -n1 "${DIR}"/dir=fwd.threads=$t.accel=0.opt-stride=1.j=1.arr=packed.workers=1.type=float.inplace=$i.txt | sed 's/^.*\t//')
		TIME=$(bc <<< "scale=${SCALE}; ${LAST} * ${UNIT}" | awk '{printf "%.1f\n", $0}')
		RATIO=$(bc <<< "scale=2; $BASE / $TIME")
		echo -en "&\t${TIME}\t&\t${RATIO}\t";
	done;
	echo "\\\\";
done
