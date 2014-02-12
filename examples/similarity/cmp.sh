#!/bin/bash

: ${TST:=Tst6}

DIR=tmp/${TST}

REFDIR=${DIR}/refImages

IMAGES=( "${REFDIR}"/* )
IMAGES=( "${IMAGES[@]##*/}" )

TESTDIR=${DIR}
TESTS=( "WL0" "WL1" "WL2" "openGLinterp" )

BIN=./compare

function compare()
{
	RET=$( "${BIN}" "$1" "$2" 2>&1 )
	PSNR=$( echo "${RET}" | grep "psnr" | sed 's/.*=//' )
	SSIM=$( echo "${RET}" | grep "ssim" | sed 's/.*=//' )
	PATCHES=$( echo "${RET}" | grep "patches" | sed 's/.*=//' )
	GLOBAL=$( echo "${RET}" | grep "global" | sed 's/.*=//' )
	echo -e "\t| $3 \t| $4   \t| ${PSNR} \t| ${SSIM} \t| ${PATCHES} \t| ${GLOBAL} \t|"
}

echo -e "\t|-------\t+-------\t+-------\t+-------\t+----------\t+---------\t|"
echo -e "\t| image \t|  test \t|  PSNR \t|  SSIM \t|  PATCHES \t|  GLOBAL \t|"
echo -e "\t|-------\t+-------\t+-------\t+-------\t+----------\t+---------\t|"

for I in "${IMAGES[@]}"; do
	for T in "${TESTS[@]}"; do
		compare "${REFDIR}"/"${I}" "${TESTDIR}"/"${T}"/"${I}" "${I}" "${T:0:3}"
	done
	echo -e "\t|-------\t+-------\t+-------\t+-------\t+----------\t+---------\t|"
done
