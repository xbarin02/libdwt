/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief PicoBlaze firmware performing 4 lifting steps and scaling before lifting together.
 */

#include "../../api/20-pb-firmware/pbbcelib.h"

int main()
{
	unsigned int steps;

	pb2mb_report_running();

	while( mbpb_exchange_data(0) )
	{
		steps = read_bce_cmem_u16(0x01, 0);

		// NOTE: C[5:2:] <= A[5:2:] * B[8:0:] (steps)
		// banks
		pb2dfu_set_bank(DFUAG_0, MBANK_C);
		pb2dfu_set_bank(DFUAG_1, MBANK_A);
		pb2dfu_set_bank(DFUAG_2, MBANK_B);
		// offsets
		pb2dfu_set_addr(DFUAG_0, 5);
		pb2dfu_set_addr(DFUAG_1, 5);
		pb2dfu_set_addr(DFUAG_2, 8);
		// increments
		pb2dfu_set_inc(DFUAG_0, 2);
		pb2dfu_set_inc(DFUAG_1, 2);
		pb2dfu_set_inc(DFUAG_2, 0);
		// count
		pb2dfu_set_cnt(steps);
		// operation
		pb2dfu_restart_op(DFU_VMUL);

		// NOTE: C[4:2:] <= A[4:2:] * B[10:0:] (steps)
		// banks
		//pb2dfu_set_bank(DFUAG_0, MBANK_C);
		//pb2dfu_set_bank(DFUAG_1, MBANK_A);
		//pb2dfu_set_bank(DFUAG_2, MBANK_B);
		// offsets
		pb2dfu_set_addr(DFUAG_0, 4);
		pb2dfu_set_addr(DFUAG_1, 4);
		pb2dfu_set_addr(DFUAG_2, 10);
		// increments
		//pb2dfu_set_inc(DFUAG_0, 2);
		//pb2dfu_set_inc(DFUAG_1, 2);
		//pb2dfu_set_inc(DFUAG_2, 0);
		// count
		//pb2dfu_set_cnt(steps);
		// operation
		pb2dfu_wait4hw();
		pb2dfu_restart_op(DFU_VMUL);

		// NOTE: C[0:1:] <= A[0:1:] (4)
		// banks
		//pb2dfu_set_bank(DFUAG_0, MBANK_C);
		//pb2dfu_set_bank(DFUAG_1, MBANK_A);
		// offsets
		pb2dfu_set_addr(DFUAG_0, 0);
		pb2dfu_set_addr(DFUAG_1, 0);
		// increments
		pb2dfu_set_inc(DFUAG_0, 1);
		pb2dfu_set_inc(DFUAG_1, 1);
		// cnd
		pb2dfu_set_cnt(4);
		// operation
		pb2dfu_wait4hw();
		pb2dfu_restart_op(DFU_VCOPY);

		// NOTE: C[1:2:] <= A[3:2:] * B[6:0:] (steps+1)
		// banks
		pb2dfu_set_bank(DFUAG_0, MBANK_A);
		pb2dfu_set_bank(DFUAG_1, MBANK_C);
		//pb2dfu_set_bank(DFUAG_2, MBANK_B);
		// offsets
		pb2dfu_set_addr(DFUAG_0, 1);
		pb2dfu_set_addr(DFUAG_1, 3);
		pb2dfu_set_addr(DFUAG_2, 6);
		// increments
		pb2dfu_set_inc(DFUAG_0, 2);
		pb2dfu_set_inc(DFUAG_1, 2);
		//pb2dfu_set_inc(DFUAG_2, 0);
		// count
		pb2dfu_set_cnt(steps+1);
		// operation
		pb2dfu_wait4hw();
		pb2dfu_restart_op(DFU_VMUL);

		// NOTE: B[1:2:] <= A[4:2:] + C[1:2:] (steps)
		// banks
		pb2dfu_set_bank(DFUAG_0, MBANK_B);
		//pb2dfu_set_bank(DFUAG_1, MBANK_C);
		pb2dfu_set_bank(DFUAG_2, MBANK_A);
		// offsets
		//pb2dfu_set_addr(DFUAG_0, 1);
		pb2dfu_set_addr(DFUAG_1, 4);
		pb2dfu_set_addr(DFUAG_2, 1);
		// increments
		//pb2dfu_set_inc(DFUAG_0, 2);
		//pb2dfu_set_inc(DFUAG_1, 2);
		pb2dfu_set_inc(DFUAG_2, 2);
		// count
		pb2dfu_set_cnt(steps);
		// operation
		pb2dfu_wait4hw();
		pb2dfu_restart_op(DFU_VADD);

		// NOTE: A[4:2:] <= B[1:2:] + C[3:2:] (steps)
		// banks
		pb2dfu_set_bank(DFUAG_0, MBANK_C);
		pb2dfu_set_bank(DFUAG_1, MBANK_B);
		//pb2dfu_set_bank(DFUAG_2, MBANK_A);
		// offsets
		pb2dfu_set_addr(DFUAG_0, 4);
		pb2dfu_set_addr(DFUAG_1, 1);
		pb2dfu_set_addr(DFUAG_2, 3);
		// increments
		//pb2dfu_set_inc(DFUAG_0, 2);
		//pb2dfu_set_inc(DFUAG_1, 2);
		//pb2dfu_set_inc(DFUAG_2, 2);
		// count
		//pb2dfu_set_cnt(steps);
		// operation
		pb2dfu_wait4hw();
		pb2dfu_restart_op(DFU_VADD);

		// NOTE: C[1:2:] <= A[2:2:] * B[4:0:] (steps+1)
		// banks
		pb2dfu_set_bank(DFUAG_0, MBANK_A);
		pb2dfu_set_bank(DFUAG_1, MBANK_C);
		pb2dfu_set_bank(DFUAG_2, MBANK_B);
		// offsets
		pb2dfu_set_addr(DFUAG_0, 1);
		pb2dfu_set_addr(DFUAG_1, 2);
		pb2dfu_set_addr(DFUAG_2, 4);
		// increments
		//pb2dfu_set_inc(DFUAG_0, 2);
		//pb2dfu_set_inc(DFUAG_1, 2);
		pb2dfu_set_inc(DFUAG_2, 0);
		// count
		pb2dfu_set_cnt(steps+1);
		// operation
		pb2dfu_wait4hw();
		pb2dfu_restart_op(DFU_VMUL);

		// NOTE: B[1:2:] <= A[3:2:] + C[1:2:] (steps)
		// banks
		pb2dfu_set_bank(DFUAG_0, MBANK_B);
		//pb2dfu_set_bank(DFUAG_1, MBANK_C);
		pb2dfu_set_bank(DFUAG_2, MBANK_A);
		// offsets
		//pb2dfu_set_addr(DFUAG_0, 1);
		pb2dfu_set_addr(DFUAG_1, 3);
		pb2dfu_set_addr(DFUAG_2, 1);
		// increments
		//pb2dfu_set_inc(DFUAG_0, 2);
		//pb2dfu_set_inc(DFUAG_1, 2);
		pb2dfu_set_inc(DFUAG_2, 2);
		// count
		pb2dfu_set_cnt(steps);
		// operation
		pb2dfu_wait4hw();
		pb2dfu_restart_op(DFU_VADD);

		// NOTE: A[3:2:] <= B[1:2:] + C[3:2:] (steps)
		// banks
		pb2dfu_set_bank(DFUAG_0, MBANK_C);
		pb2dfu_set_bank(DFUAG_1, MBANK_B);
		//pb2dfu_set_bank(DFUAG_2, MBANK_A);
		// offsets
		pb2dfu_set_addr(DFUAG_0, 3);
		pb2dfu_set_addr(DFUAG_1, 1);
		pb2dfu_set_addr(DFUAG_2, 3);
		// increments
		//pb2dfu_set_inc(DFUAG_0, 2);
		//pb2dfu_set_inc(DFUAG_1, 2);
		//pb2dfu_set_inc(DFUAG_2, 2);
		// count
		//pb2dfu_set_cnt(steps);
		// operation
		pb2dfu_wait4hw();
		pb2dfu_restart_op(DFU_VADD);

		// NOTE: C[1:2:] <= A[1:2:] * B[2:0:] (steps+1)
		// banks
		pb2dfu_set_bank(DFUAG_0, MBANK_A);
		pb2dfu_set_bank(DFUAG_1, MBANK_C);
		pb2dfu_set_bank(DFUAG_2, MBANK_B);
		// offsets
		pb2dfu_set_addr(DFUAG_0, 1);
		//pb2dfu_set_addr(DFUAG_1, 1);
		pb2dfu_set_addr(DFUAG_2, 2);
		// increments
		//pb2dfu_set_inc(DFUAG_0, 2);
		//pb2dfu_set_inc(DFUAG_1, 2);
		pb2dfu_set_inc(DFUAG_2, 0);
		// count
		pb2dfu_set_cnt(steps+1);
		// operation
		pb2dfu_wait4hw();
		pb2dfu_restart_op(DFU_VMUL);

		// NOTE: B[1:2:] <= A[2:2:] + C[1:2:] (steps)
		// banks
		pb2dfu_set_bank(DFUAG_0, MBANK_B);
		//pb2dfu_set_bank(DFUAG_1, MBANK_C);
		pb2dfu_set_bank(DFUAG_2, MBANK_A);
		// offsets
		//pb2dfu_set_addr(DFUAG_0, 1);
		pb2dfu_set_addr(DFUAG_1, 2);
		pb2dfu_set_addr(DFUAG_2, 1);
		// increments
		//pb2dfu_set_inc(DFUAG_0, 2);
		//pb2dfu_set_inc(DFUAG_1, 2);
		pb2dfu_set_inc(DFUAG_2, 2);
		// count
		pb2dfu_set_cnt(steps);
		// operation
		pb2dfu_wait4hw();
		pb2dfu_restart_op(DFU_VADD);

		// NOTE: A[2:2:] <= B[1:2:] + C[3:2:] (steps)
		// banks
		pb2dfu_set_bank(DFUAG_0, MBANK_C);
		pb2dfu_set_bank(DFUAG_1, MBANK_B);
		//pb2dfu_set_bank(DFUAG_2, MBANK_A);
		// offsets
		pb2dfu_set_addr(DFUAG_0, 2);
		pb2dfu_set_addr(DFUAG_1, 1);
		pb2dfu_set_addr(DFUAG_2, 3);
		// increments
		//pb2dfu_set_inc(DFUAG_0, 2);
		//pb2dfu_set_inc(DFUAG_1, 2);
		//pb2dfu_set_inc(DFUAG_2, 2);
		// count
		//pb2dfu_set_cnt(steps);
		// operation
		pb2dfu_wait4hw();
		pb2dfu_restart_op(DFU_VADD);

		// NOTE: C[1:2:] <= A[0:2:] * B[0:0:] (steps+1)
		// banks
		pb2dfu_set_bank(DFUAG_0, MBANK_A);
		pb2dfu_set_bank(DFUAG_1, MBANK_C);
		pb2dfu_set_bank(DFUAG_2, MBANK_B);
		// offsets
		pb2dfu_set_addr(DFUAG_0, 1);
		pb2dfu_set_addr(DFUAG_1, 0);
		pb2dfu_set_addr(DFUAG_2, 0);
		// increments
		//pb2dfu_set_inc(DFUAG_0, 2);
		//pb2dfu_set_inc(DFUAG_1, 2);
		pb2dfu_set_inc(DFUAG_2, 0);
		// count
		pb2dfu_set_cnt(steps+1);
		// operation
		pb2dfu_wait4hw();
		pb2dfu_restart_op(DFU_VMUL);

		// NOTE: B[1:2:] <= A[1:2:] + C[1:2:] (steps)
		// banks
		pb2dfu_set_bank(DFUAG_0, MBANK_B);
		//pb2dfu_set_bank(DFUAG_1, MBANK_C);
		pb2dfu_set_bank(DFUAG_2, MBANK_A);
		// offsets
		//pb2dfu_set_addr(DFUAG_0, 1);
		pb2dfu_set_addr(DFUAG_1, 1);
		pb2dfu_set_addr(DFUAG_2, 1);
		// increments
		//pb2dfu_set_inc(DFUAG_0, 2);
		//pb2dfu_set_inc(DFUAG_1, 2);
		pb2dfu_set_inc(DFUAG_2, 2);
		// count
		pb2dfu_set_cnt(steps);
		// operation
		pb2dfu_wait4hw();
		pb2dfu_restart_op(DFU_VADD);

		// NOTE: A[1:2:] <= B[1:2:] + C[3:2:] (steps)
		// banks
		pb2dfu_set_bank(DFUAG_0, MBANK_C);
		pb2dfu_set_bank(DFUAG_1, MBANK_B);
		//pb2dfu_set_bank(DFUAG_2, MBANK_A);
		// offsets
		//pb2dfu_set_addr(DFUAG_0, 1);
		//pb2dfu_set_addr(DFUAG_1, 1);
		pb2dfu_set_addr(DFUAG_2, 3);
		// increments
		//pb2dfu_set_inc(DFUAG_0, 2);
		//pb2dfu_set_inc(DFUAG_1, 2);
		//pb2dfu_set_inc(DFUAG_2, 2);
		// count
		//pb2dfu_set_cnt(steps);
		// operation
		pb2dfu_wait4hw();
		pb2dfu_restart_op(DFU_VADD);

		pb2dfu_wait4hw();
		mbpb_exchange_data(0);
	}

	mbpb_exchange_data(0);

	while (1)
		;
}
