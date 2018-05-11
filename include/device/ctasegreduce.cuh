/******************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 *
 * Code and text by Sean Baxter, NVIDIA Research
 * See http://nvlabs.github.io/moderngpu for repository and documentation.
 *
 ******************************************************************************/

#pragma once

#include "ctasegscan.cuh"
#include "ctasearch.cuh"

//#include "../constants.h"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// Segmented reduce utility functions.

// Extract the upper-bound indices from the coded ranges. Decrement to include 
// the first addressed row/segment.

struct SegReduceRange {
	int begin;
	int end;
	int total;
	bool flushLast;
};

MGPU_DEVICE SegReduceRange DeviceShiftRange(int limit0, int limit1) {
	SegReduceRange range;
	range.begin = 0x7fffffff & limit0;
	range.end = 0x7fffffff & limit1; 
	range.total = range.end - range.begin;
	range.flushLast = 0 == (0x80000000 & limit1);
	range.end += !range.flushLast;
	return range;
}

// Reconstitute row/segment indices from a starting row index and packed end 
// flags. Used for pre-processed versions of interval reduce and interval Spmv.
template<int VT>
MGPU_DEVICE void DeviceExpandFlagsToRows(int first, int endFlags, 
	int rows[VT + 1]) {

	rows[0] = first;
	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		if((1<< i) & endFlags) ++first;
		rows[i + 1] = first;
	}
}

////////////////////////////////////////////////////////////////////////////////
// After loading CSR terms into shared memory, each thread binary searches 
// (upper-bound) to find its starting point. Each thread then walks forward,
// emitting the csr0-relative row indices to register.

template<int NT, int VT>
MGPU_DEVICE int DeviceExpandCsrRows(int tidOffset, int offset,
  const int* csr_shared, int numRows, int end, int rows[VT + 1], 
  int rowStarts[VT]) {
		
	// Each thread binary searches for its starting row.
	int row = BinarySearch<MgpuBoundsUpper>(csr_shared, numRows, offset,
		mgpu::less<int>()) - 1;

	// Each thread starts at row and scans forward, emitting row IDs into
	// register. Store the CTA-local row index (starts at 0) to rows and the
	// start of the row (globally) to rowStarts.
	int curOffset = csr_shared[row];
	int nextOffset = (row + 1 < numRows) ? csr_shared[row + 1] : end;

	rows[0] = row;
	rowStarts[0] = curOffset;
	int endFlags = 0;
  //if( threadIdx.x<32 && blockIdx.z==0 )
  //printf("tid:%d,row:%d\n", threadIdx.x, row);
	
	#pragma unroll
	for(int i = 1; i <= VT; ++i) {
		// Advance the row cursor when the iterator hits the next row offset.
		if( offset + i == nextOffset) {
			// Set an end flag when the cursor advances to the next row.
			endFlags |= 1<< (i - 1);

			// Advance the cursor and load the next row offset.
			++row;
			curOffset = nextOffset;
			nextOffset = (row + 1 < numRows) ? csr_shared[row + 1] : end;
		}
		rows[i] = row;
		if(i < VT) rowStarts[i] = curOffset;
	}
	__syncthreads();

	return endFlags;
}

////////////////////////////////////////////////////////////////////////////////
// DeviceSegReducePrepare
// Expand non-empty interval of CSR elements into row indices. Compute end-flags
// by comparing adjacent row IDs.

// DeviceSegReducePrepare may be called either by a pre-processing kernel or by
// the kernel that actually evaluates the segmented reduction if no preprocesing
// is desired.
struct SegReduceTerms {
	int endFlags;
	int tidDelta;
};

template<int NT, int VT>
MGPU_DEVICE SegReduceTerms DeviceSegReducePrepare(int* csr_shared, int numRows, 
	int tid, int gid, bool flushLast, int rows[VT + 1], int rowStarts[VT]) {

	// Pass a sentinel (end) to point to the next segment start. If we flush,
	// this is the end of this tile. Otherwise it is INT_MAX
	int endFlags = DeviceExpandCsrRows<NT, VT>(gid + VT * tid, csr_shared,
		numRows, flushLast ? (gid + NT * VT) : INT_MAX, rows, rowStarts);

	// Find the distance to to scan to compute carry-in for each thread. Use the
	// existance of an end flag anywhere in the thread to determine if carry-out
	// values from the left should propagate through to the right.
	int tidDelta = DeviceFindSegScanDelta<NT>(tid, rows[0] != rows[VT],
		csr_shared);

	SegReduceTerms terms = { endFlags, tidDelta };
	return terms;
}

template<int NT, int VT>
MGPU_DEVICE SegReduceTerms DeviceSegReducePrepareSpmm(const int* csr_shared, 
    int* csr_shared2, int numRows, int offset, int tid, int gid, 
    bool flushLast, int rows[VT + 1], int rowStarts[VT]) {

	// Pass a sentinel (end) to point to the next segment start. If we flush,
	// this is the end of this tile. Otherwise it is INT_MAX
	int endFlags = DeviceExpandCsrRows<NT, VT>(gid + VT * tid, gid+VT*(tid%32)+offset, csr_shared,
		numRows, flushLast ? (gid + NT * VT) : INT_MAX, rows, rowStarts);

	// Find the distance to to scan to compute carry-in for each thread. Use the
	// existance of an end flag anywhere in the thread to determine if carry-out
	// values from the left should propagate through to the right.
	int tidDelta = DeviceFindSegScanDelta<NT>(tid, rows[0] != rows[VT],
		csr_shared2);

	SegReduceTerms terms = { endFlags, tidDelta };
	return terms;
}
////////////////////////////////////////////////////////////////////////////////
// CTASegReduce
// Core segmented reduction code. Supports fast-path and slow-path for intra-CTA
// segmented reduction. Stores partials to global memory.
// Callers feed CTASegReduce::ReduceToGlobal values in thread order.
template<int NT, int TB, bool HalfCapacity, typename T, typename Op>
struct CTASegReduce {
	typedef CTASegScan<NT, Op> SegScan;

	enum {
		NV = NT,
		Capacity = HalfCapacity ? (NV / 2) : NV
	};

	union Storage {
		typename SegScan::Storage segScanStorage;
		T values[Capacity];
		//T values[Capacity*TB];
	};
	
	template<typename DestIt>
	MGPU_DEVICE static void ReduceToGlobalSpmm(
    const int rows[TB + 1], int total, int tidDelta, int startRow, int block, 
    int tid, int lane_id, T data[TB], int B_ncols, DestIt dest_global, 
    T* carryOut_global, int slab, T identity, Op op, T* storage)   {

		// Run a segmented scan within the thread.
		T x, localScan[TB];
    //#pragma unroll
    for(int i = 0; i < TB; ++i)
    {
      x = i ? op(x, data[i]) : data[i];
      localScan[i] = x;
      //if( tid+blockIdx.x*blockDim.x==7 ) printf("%d: %d %d %d %f\n", blockIdx.z, i, rows[i], rows[i+1], x);
      if(rows[i] != rows[i + 1]) 
        x = identity;
    }

    T carryIn, carryOut;
    //if( tid<16 && blockIdx.z==0 ) printf("tid:%d: %f,%f\n", tid, carryOut, carryInPrev);
    storage[tid] = carryOut;
    __syncthreads();

    //if( x!=0.f )
    //  printf("block:%d, tid:%d, tidDelta:%d, x:%f\n", blockIdx.z, tid, tidDelta, x);

		// Run an inclusive scan 
		int first = 0;
		storage[first + tid] = x;
		__syncthreads();

		#pragma unroll
		for(int offset = 32; offset < NT; offset += offset) {
			if(tidDelta >= offset) 
				x = op(storage[first + tid - offset], x);
			first = NT - first;
			storage[first + tid] = x;
			__syncthreads();
		}

		// Get the exclusive scan.
		x = (tid>=32) ? storage[first + tid - 32] : identity;
		carryOut = storage[first + NT - 32 + lane_id];

		__syncthreads();
    carryIn = x;
    //if( blockIdx.x==0 )
    //  printf("block.z:%d, tid:%d, x:%d\n", blockIdx.z, tid, x);

    // Store carryOut to global memory
    if(tid<32) carryOut_global[block*B_ncols+tid] = carryOut;

		// Run a parallel segmented scan over the carry-out values to compute
		// carry-in.
		dest_global += startRow*B_ncols;

		// Add carry-in to each thread-local scan value. Store directly
		// to global.
		#pragma unroll
		for(int i = 0; i < TB; ++i) {
			// Add the carry-in to the local scan.
			T x2 = op(carryIn, localScan[i]);

			// Store on the end flag and clear the carry-in.
			if(rows[i] != rows[i + 1]) {
				carryIn = identity;
				dest_global[(rows[i]*B_ncols)+lane_id] = x2;
			}
		}
 
    // If not first warp, read from carryOut
    //if( tid>=32 )

    // If not last warp, write to dest_global
    /*if( tid<NT-32 && rows[TB-1]==rows[TB] )
      dest_global[(rows[TB]*B_ncols)+lane_id] += carryOut;

    // If last warp of last block
    //if( blockIdx.x==blockDim.x-1 )
    if( tid>=NT-32 )
      carryOut_global[(block*B_ncols)+lane_id] = carryOut;
      //if( carryOut[j]>0.f ) printf("%d:%f\n", tid, carryOut[j]);*/

	  /*__syncthreads();

  	// Cooperatively store reductions to global memory.
		for(int index = tid; index < total; index += NT)
			dest_global[index] = storage.values[index];
		__syncthreads();*/
	}
};

} // namespace mgpu
