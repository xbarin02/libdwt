#!/bin/bash

: ${TST:=Tst6}

DIR=tmp/${TST}

REFDIR=${DIR}/refImages

IMAGES=( "${REFDIR}"/* )
IMAGES=( "${IMAGES[@]##*/}" )

TESTDIR=${DIR}
TESTS=( "WL0" "WL1" "WL2" )

BIN=./compare

declare -A DIR
DIR[PSNR]=\<
DIR[SSIM]=\<
DIR[PATCHES]=\>

function compare()
{
	local RET=$( "${BIN}" "$1" "$2" 2>&1 )
	local PSNR=$( echo "${RET}" | grep "psnr" | sed 's/.*=//' )
	local SSIM=$( echo "${RET}" | grep "ssim" | sed 's/.*=//' )
	local PATCHES=$( echo "${RET}" | grep "patches" | sed 's/.*=//' )
	local GLOBAL=$( echo "${RET}" | grep "global" | sed 's/.*=//' )

	local TEST=${3}
	TEST=${TEST/.png/}

	local WAVELET=$( echo "$4" | sed 's/WL0/CDF 9\/7/;s/WL1/CDF 5\/3/;s/WL2/bi-linear/' )

	declare -A PRE
	for METRIC in PSNR SSIM PATCHES; do
		eval PRE[${METRIC}]=""
	done

	local M=$5
	local TMP
	local METRIC
	local IS_MAX
	for METRIC in PSNR SSIM PATCHES; do
		eval TMP=\"\${${M}[${METRIC}]} == \${${METRIC}}\"
		IS_MAX=$(bc <<< "${TMP}")
		if [ "${IS_MAX}" == "1" ]; then
			eval PRE[${METRIC}]=\\\\\\\\bf
		fi
	done

	echo -e "\t\t\t${TEST}\t& ${WAVELET}\t& {${PRE[SSIM]}${SSIM}}\t\t& {${PRE[PATCHES]}${PATCHES}} \\\\\\\\"
}

function getmax()
{
	local RET=$( "${BIN}" "$1" "$2" 2>&1 )
	local PSNR=$( echo "${RET}" | grep "psnr" | sed 's/.*=//' )
	local SSIM=$( echo "${RET}" | grep "ssim" | sed 's/.*=//' )
	local PATCHES=$( echo "${RET}" | grep "patches" | sed 's/.*=//' )
	local GLOBAL=$( echo "${RET}" | grep "global" | sed 's/.*=//' )
	local M=$3
	local TMP

	local METRIC
	for METRIC in PSNR SSIM PATCHES; do
		eval TMP=\"\${${M}[${METRIC}]} \${DIR[$METRIC]} \${${METRIC}}\"
		local IS_MAX=$(bc <<< "${TMP}")
		if [ "${IS_MAX}" == "1" ]; then
			eval $M[${METRIC}]=\$${METRIC}
		fi
	done
}

echo -e "% ${TST}"
echo -e "\\\\begin{table}"
echo -e "\t\\\\centering"
echo -e "\t\\\\begin{tabular}{|l|l|r|r|}"
echo -e "\t\t\hline"
echo -e "\t\t\t{\\\\bf image} & {\\\\bf interpolation} & {\\\\bf SSIM} & {\\\\bf PBC} \\\\\\\\"
echo -e "\t\t\hline"

for I in "${IMAGES[@]}"; do
	declare -A MAX
	MAX[PSNR]=0
	MAX[SSIM]=0
	MAX[PATCHES]=99
	for T in "${TESTS[@]}"; do
		getmax "${REFDIR}"/"${I}" "${TESTDIR}"/"${T}"/"${I}" MAX
	done
	#echo "MAX: PSNR=${MAX[PSNR]} SSIM=${MAX[SSIM]} PATCHES=${MAX[PATCHES]}"
	for T in "${TESTS[@]}"; do
		compare "${REFDIR}"/"${I}" "${TESTDIR}"/"${T}"/"${I}" "${I}" "${T:0:3}" MAX
	done
	echo -e "\t\t\hline"
done

echo -e "\t\\\\end{tabular}"
echo -e "\t\\\\caption{Results for scene ${TST}.}"
echo -e "\t\\\\label{tbl:results-${TST}}"
echo -e "\\\\end{table}"
echo -e ""
