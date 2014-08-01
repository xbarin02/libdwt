#!/bin/bash

# directories
SDIR=../train-test
TMPDIR=.
OUTDIR=./tmp

rm -rf -- "${OUTDIR}"
mkdir -- "${OUTDIR}"
cd -- "${OUTDIR}"

# hole program (or gnuplot)
HOLE=../hole

if [[ ! -x "${HOLE}" ]]; then
    echo "File '"${HOLE}"' is not executable or found"
    exit
fi

# aux. files
TMPFILE=${TMPDIR}/grid-temp
RESFILE=${TMPDIR}/results
DURFILE=${TMPDIR}/time-temp

rm -f -- "${RESFILE}"

for f in "${SDIR}"/fv_*.train.svm; do
	#echo "processing ${f}..."

	COEFFS=$( cat "${f}" | head -n1 | sed -r 's/[[:blank:]]+/ /g' | tr ' ' '\n' | tail -n1 | cut -d: -f1 )
	FILE=$(basename "${f/.train.svm/}")
	TIMEFORMAT=%0R
	( time svm-grid -gnuplot "${HOLE}" "${f}" 2>&1 | tail -n1 ) 2> "${DURFILE}" > "${TMPFILE}"
	RESULT_C=$(cat <"${TMPFILE}" | cut -d' ' -f1)
	LOG_C=$(echo "l(${RESULT_C/[eE]/*10^})/l(2)" | bc -l | sed 's/\..*//')
	RESULT_G=$(cat <"${TMPFILE}" | cut -d' ' -f2)
	LOG_G=$(echo "l(${RESULT_G/[eE]/*10^})/l(2)" | bc -l | sed 's/\..*//')
	RESULT_R=$(cat <"${TMPFILE}" | cut -d' ' -f3)
	TIME_S=$(<"${DURFILE}")
	TIME_M2=$(echo "${TIME_S}/60" | bc -l | sed -r 's/([[:digit:]]*\.[[:digit:]]{2}).*/\1/')

	svm-train -c ${RESULT_C} -g ${RESULT_G} "${f}" "${f}.model" > /dev/null # FIXME: c g
	TEST=$( svm-predict "${f/.train.svm/.test.svm}" "${f}.model" "${f/.train.svm/.test.svm}.predict" |& grep "Accuracy" | sed -r 's/.*= ([[:digit:]]*\.[[:digit:]]*)%.*/\1/' )

	printf "| %-24s | %6s | %18s | %7s | %7s | %-8s | %-8s |\n" "${FILE}" "${COEFFS}" "${TIME_M2}" "${LOG_C}" "${LOG_G}" "${RESULT_R}" "${TEST}"
	printf "${TEST} #| %-24s | %6s | %18s | %7s | %7s | %-16s | %-15s |\n" "${FILE}" "${COEFFS}" "${TIME_M2}" "${LOG_C}" "${LOG_G}" "${RESULT_R}" "${TEST}" >> "${RESFILE}"
done

echo

echo -e "-------------------------------------------------------------------------------------------------------------------"
echo -e "| file                     | coeffs | grid search [min.] | log2(c) | log2(g) | rate [%] (train) | rate [%] (test) |"
echo -e "-------------------------------------------------------------------------------------------------------------------"

sort -nr "${RESFILE}" | sed 's/^.*#//'

echo -e "-------------------------------------------------------------------------------------------------------------------"
