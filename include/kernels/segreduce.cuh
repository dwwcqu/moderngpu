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

#include "../kernels/csrtools.cuh"

//#include "../constants.h"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// SegReducePreprocess
	
struct SegReducePreprocessData {
	int count, numSegments, numSegments2;
	int numBlocks;
	MGPU_MEM(int) limitsDevice;
	MGPU_MEM(int) threadCodesDevice;

	// If csr2Device is set, use BulkInsert to finalize results into 
	// dest_global.
	MGPU_MEM(int) csr2Device;
};

// Generic function for prep
template<typename Tuning, typename CsrIt>
MGPU_HOST void SegReducePreprocess(int count, CsrIt csr_global, int numSegments,
	bool supportEmpty, std::auto_ptr<SegReducePreprocessData>* ppData, 
	CudaContext& context) {

	std::auto_ptr<SegReducePreprocessData> data(new SegReducePreprocessData);

	int2 launch = Tuning::GetLaunchParams(context);
	int NV = launch.x * launch.y;

	int numBlocks = MGPU_DIV_UP(count, NV);
	data->count = count;
	data->numSegments = data->numSegments2 = numSegments;
	data->numBlocks = numBlocks;

	// Filter out empty rows and build a replacement structure.
	if(supportEmpty) {
		MGPU_MEM(int) csr2Device = context.Malloc<int>(numSegments + 1);
		CsrStripEmpties<false>(count, csr_global, (const int*)0, numSegments,
			csr2Device->get(), (int*)0, (int*)&data->numSegments2, context); 
		if(data->numSegments2 < numSegments) {
			csr_global = csr2Device->get();
			numSegments = data->numSegments2;
			data->csr2Device = csr2Device;
		}
	}

	data->limitsDevice = PartitionCsrSegReduce(count, NV, csr_global,
		numSegments, (const int*)0, numBlocks + 1, context);
	data->threadCodesDevice = BuildCsrPlus<Tuning>(count, csr_global, 
		data->limitsDevice->get(), numBlocks, context);

	*ppData = data;
}

////////////////////////////////////////////////////////////////////////////////
// SegReduceSpine
// Compute the carry-in in-place. Return the carry-out for the entire tile.
// A final spine-reducer scans the tile carry-outs and adds into individual
// results.

template<int NT, typename T, typename DestIt, typename Op>
__global__ void KernelSegReduceSpine1(const int* limits_global, 
  int count, DestIt dest_global, const T* carryIn_global, T identity, Op op,
	T* carryOut_global) {

	typedef CTASegScan<NT, Op> SegScan;
	union Shared {
		typename SegScan::Storage segScanStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
	int block = blockIdx.x;
	int gid = NT * block + tid;

	// Load the current carry-in and the current and next row indices.
	int row = (gid < count) ? 
		(0x7fffffff & limits_global[gid]) :
		INT_MAX;
	int row2 = (gid + 1 < count) ? 
		(0x7fffffff & limits_global[gid + 1]) :
		INT_MAX;

  // Depending on register usage, could consider changing MGPU_TB 
  // here to MGPU_BC
	/*T carryIn2[MGPU_TB];
	T dest[    MGPU_TB];

	// Run a segmented scan of the carry-in values.
	bool endFlag = row != row2;

	T carryOut[MGPU_TB];
	T x[       MGPU_TB];

  for( int slab=0; slab<MGPU_BC/MGPU_TB; slab++ )
  {
    #pragma unroll
    for( int j=0; j<MGPU_TB; j++ )
    {	
      carryIn2[j] = (gid < count) ? 
        carryIn_global[block+j*gridDim.x+slab*MGPU_TB*gridDim.x] : identity;
      dest[j]     = (gid < count) ? 
        dest_global[row*MGPU_BC+j+slab*MGPU_TB] : identity;
      x[j] = SegScan::SegScan(tid, carryIn2[j], endFlag, 
        shared.segScanStorage, &carryOut[j], identity, op);
    }

    // Store the reduction at the end of a segment to dest_global.
    if(endFlag)
      #pragma unroll
      for( int j=0; j<MGPU_TB; j++ )
        dest_global[row*MGPU_BC+j+slab*MGPU_TB] = op(x[j], dest[j]);
    
    // Store the CTA carry-out.
    if(!tid)
      #pragma unroll
      for( int j=0; j<MGPU_TB; j++ )
        carryOut_global[block+j*gridDim.x+slab*MGPU_TB*gridDim.x] = carryOut[j];
  }*/
}

////////////////////////////////////////////////////////////////////////////////
// DeviceFindSegScanDelta
// Runs an inclusive max-index scan over binary inputs.

template<int NT>
MGPU_DEVICE int DeviceFindSegScanDeltaSpmm(int tid, bool flag, 
    int* delta_shared) {
	const int NumWarps = NT / 32;

	int warp = tid / 32;
	int lane = 31 & tid;
	uint warpMask = 0xffffffff>> (31 - lane);		// inclusive search
	uint ctaMask = 0x7fffffff>> (31 - lane);		// exclusive search

	uint warpBits = __ballot(flag);
	delta_shared[warp] = warpBits;
	__syncthreads();

	if(tid < NumWarps) {
		uint ctaBits = __ballot(0 != delta_shared[tid]);
		int warpSegment = 31 - clz(ctaMask & ctaBits);
		int start = (-1 != warpSegment) ? 
			(31 - clz(delta_shared[warpSegment]) + 32 * warpSegment) : 0;
		delta_shared[NumWarps + tid] = start;
	}
	__syncthreads();

	// Find the closest flag to the left of this thread within the warp.
	// Include the flag for this thread.
	int start = 31 - clz(warpMask & warpBits);
	if(-1 != start) start += ~31 & tid;
	else start = delta_shared[NumWarps + warp];
	__syncthreads();

	return tid - start;
}

template<int TB, int NT, typename T, typename DestIt, typename Op>
__global__ void KernelSegReduceSpine1Prealloc(const int* limits_global, 
  int count, DestIt dest_global, const T* carryIn_global, T identity, Op op,
	T* carryOut_global, int B_ncols) {

	typedef CTASegScan<NT, Op> SegScan;
	union Shared {
		typename SegScan::Storage segScanStorage;
	};
	__shared__ Shared shared;

	int tid     = threadIdx.x;
	int block   = blockIdx.x;
	int gid     = NT * block + tid;
  int warp_id = tid >> 5;
  int glob_id = gid >> 5;
  int lane_id = tid & (32-1);

	// Load the current carry-in and the current and next row indices.
	int row = (glob_id < count) ? 
		(0x7fffffff & limits_global[glob_id]) :
		INT_MAX;
	int row2 = (glob_id + 1 < count) ? 
		(0x7fffffff & limits_global[glob_id + 1]) :
		INT_MAX;

	T carryIn2 = (glob_id < count) ? carryIn_global[glob_id*B_ncols+lane_id+(blockIdx.z<<5)] 
                                 : identity;
  T dest     = (glob_id < count) ? 
               dest_global[row*B_ncols+lane_id+(blockIdx.z<<5)] : identity;

	// Run a segmented scan of the carry-in values.
	bool endFlag = row != row2;

  //if( blockIdx.x<2 && carryIn2>0.f )
  //  printf("spine1 %d %d: %d %d %f %f\n", blockIdx.z, gid, row, row2, carryIn2, dest);

	T carryOut;
  T x = carryIn2;

  //int tidDelta = DeviceFindSegScanDeltaSpmm<NT>(tid, endFlag, 
  //    shared.segScanStorage.delta);
  int tidDelta = 0;

  if( lane_id==0 )
    shared.segScanStorage.delta[warp_id] = endFlag;
  __syncthreads();

  if( lane_id==0 ) {
    if( !endFlag )
      for( int i=1; i<=warp_id; i++ ) {
        if( shared.segScanStorage.delta[warp_id-i] ) {
          tidDelta = (i-1)<<5;
          break;
        }
      }
  }

  tidDelta = __shfl(tidDelta, 0);

  //if( glob_id<8 ) printf("tidDelta1 %d %d: %d\n", blockIdx.z, tid, tidDelta);

	int first = 0;
	shared.segScanStorage.values[first + tid] = x;
	__syncthreads();

	#pragma unroll
	for(int offset = 32; offset < NT; offset += offset) {
		if(tidDelta >= offset) 
			x = op(shared.segScanStorage.values[first + tid - offset], x);
		first = NT - first;
		shared.segScanStorage.values[first + tid] = x;
		__syncthreads();
	}

	// Get the exclusive scan.
	x = (tid>=32) ? shared.segScanStorage.values[first + tid - 32] : identity;
	carryOut = shared.segScanStorage.values[first + NT - 32 + lane_id];
	__syncthreads();
			
	// Store the reduction at the end of a segment to dest_global.
	if(endFlag)
		dest_global[row*B_ncols+lane_id+(blockIdx.z<<5)] = op(x, dest);
	
	// Store the CTA carry-out.
  if(tid<32) carryOut_global[block*B_ncols+lane_id+(blockIdx.z<<5)] = carryOut;
}

template<int NT, typename T, typename DestIt, typename Op>
__global__ void KernelSegReduceSpine2(const int* limits_global, int numBlocks,
	int count, int nv, DestIt dest_global, const T* carryIn_global, T identity,
	Op op) {

	typedef CTASegScan<NT, Op> SegScan;
	struct Shared {
		typename SegScan::Storage segScanStorage;
		int carryInRow;
		T   carryIn;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
	
	for(int i = 0; i < numBlocks; i += NT) {
		int gid = (i + tid) * nv;

		// Load the current carry-in and the current and next row indices.
		int row = (gid < count) ? 
			(0x7fffffff & limits_global[gid]) : INT_MAX;
		int row2 = (gid + nv < count) ? 
			(0x7fffffff & limits_global[gid + nv]) : INT_MAX;
		T carryIn2, dest;
		T carryOut, x;

		// Run a segmented scan of the carry-in values.
		/*bool endFlag = row != row2;

    for( int slab=0; slab<MGPU_BC/MGPU_TB; slab++ )
    {
      #pragma unroll
      for( int j=0; j<MGPU_TB; j++ )
      {
        carryIn2[j] = (i + tid < numBlocks) ? 
          carryIn_global[i+tid+j*gridDim.x+slab*MGPU_TB*gridDim.x]:identity;
        dest[j] = (gid < count) ? 
          dest_global[row*MGPU_BC+j+slab*MGPU_TB] : identity;
        x[j] = SegScan::SegScan(tid, carryIn2[j], endFlag, 
          shared.segScanStorage,&carryOut[j], identity, op);
      }

      // Add the carry-in to the reductions when we get to the end of a segment.
      if(endFlag) {
        // Add the carry-in from the last loop iteration to the carry-in
        // from this loop iteration.
        #pragma unroll
        for( int j=0; j<MGPU_TB; j++ )
        {
          if(i && row == shared.carryInRow) 
            x[j] = op(shared.carryIn[j], x[j]);
          dest_global[row*MGPU_BC+j+slab*MGPU_TB] = op(x[j], dest[j]);
        }
      }

      // Set the carry-in for the next loop iteration.
      if(i + NT < numBlocks) {
        __syncthreads();
        if(i > 0) {
          // Add in the previous carry-in.
          if(NT - 1 == tid) {
            #pragma unroll
            for( int j=0; j<MGPU_TB; j++ )
            {
              shared.carryIn[j] = (shared.carryInRow == row2) ?
                op(shared.carryIn[j], carryOut[j]) : carryOut[j];
              shared.carryInRow = row2;
            }
          }
        } else {
          if(NT - 1 == tid) {
            #pragma unroll
            for( int j=0; j<MGPU_TB; j++ )
            {
              shared.carryIn[j] = carryOut[j];
              shared.carryInRow = row2;
            }
          }
        }
        __syncthreads();
      }
    }*/
	}
}

template<int TB, int NT, typename T, typename DestIt, typename Op>
__global__ void KernelSegReduceSpine2Prealloc(const int* limits_global, 
  int numBlocks, int count, DestIt dest_global, const T* carryIn_global,
  T identity, int B_ncols, Op op) {

	typedef CTASegScan<NT, Op> SegScan;
	struct Shared {
		typename SegScan::Storage segScanStorage;
		int carryInRow;
		T carryIn[32];
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane_id = tid & (32-1);
	
	for(int i = 0; i < numBlocks; i += 4) {
		int gid = (i + warp_id) * 4;

		// Load the current carry-in and the current and next row indices.
		int row = (gid < count) ? 
			(0x7fffffff & limits_global[gid]) : INT_MAX;
		int row2 = (gid + 4 < count) ? 
			(0x7fffffff & limits_global[gid + 4]) : INT_MAX;
		T carryIn2 = (i + warp_id < numBlocks) ? 
        carryIn_global[(i+warp_id)*B_ncols+lane_id+(blockIdx.z<<5)] : identity;
    T dest = (gid < count) ? 
        dest_global[row*B_ncols+lane_id+(blockIdx.z<<5)] : identity;

		// Run a segmented scan of the carry-in values.
		bool endFlag = row != row2;

    //if( i<3 && (carryIn2>0.f || dest>0.f) ) printf("spine2 %d %d: %d %d %d %f %f\n", blockIdx.z, gid, tid, row, row2, carryIn2, dest);

		T carryOut;
		T x = carryIn2;

    // TODO: fix tidDelta. The last problem that needs to be fixed!
		//int tidDelta = DeviceFindSegScanDeltaSpmm<NT>(tid, endFlag,
		//		shared.segScanStorage.delta);

    int tidDelta = 0;

    if( lane_id==0 )
      shared.segScanStorage.delta[warp_id] = endFlag;
    __syncthreads();

    if( lane_id==0 ) {
      if( !endFlag )
        for( int i=1; i<=warp_id; i++ ) {
          if( shared.segScanStorage.delta[warp_id-i] ) {
            tidDelta = (i-1)<<5;
            break;
          }
        }
    }

    tidDelta = __shfl(tidDelta, 0);

    //if( gid==16 ) printf("tidDelta2 %d %d: %d\n", blockIdx.z, tid, tidDelta);

		int first = 0;
		shared.segScanStorage.values[first + tid] = x;
		__syncthreads();

		#pragma unroll
		for(int offset = 32; offset < NT; offset += offset) {
			if(tidDelta >= offset)
				x = op(shared.segScanStorage.values[first + tid - offset], x);
			first = NT - first;
			shared.segScanStorage.values[first + tid] = x;
			__syncthreads();
		}

		// Get the exclusive scan.
		x = (tid>=32) ? shared.segScanStorage.values[first + tid - 32] : identity;
		carryOut = shared.segScanStorage.values[first + NT - 32 + lane_id];
		__syncthreads();

		// Add the carry-in to the reductions when we get to the end of a segment.
		if(endFlag) {
			// Add the carry-in from the last loop iteration to the carry-in
			// from this loop iteration.
			if(i && row == shared.carryInRow) 
				x = op(shared.carryIn[lane_id], x);
			dest_global[row*B_ncols+lane_id+(blockIdx.z<<5)] = op(x, dest);
		}

		// Set the carry-in for the next loop iteration.
		if(i + 4 < numBlocks) {
			__syncthreads();
			if(i > 0) {
				// Add in the previous carry-in.
				if(tid >= 96) {
					shared.carryIn[lane_id] = (shared.carryInRow == row2) ?
						op(shared.carryIn[lane_id], carryOut) : carryOut;
          if( lane_id==0 )
					  shared.carryInRow = row2;
				}
			} else {
				if(tid >= 96) {
					shared.carryIn[lane_id] = carryOut;
          //if( i<2 ) printf("out spine2 %d %d: %d %d %f %f\n", blockIdx.z, gid, row, row2, carryIn2, carryOut);
          if( lane_id==0 )
					shared.carryInRow = row2;
				}
			}
			__syncthreads();
    }
	}
}

template<typename T, typename Op, typename DestIt>
MGPU_HOST void SegReduceSpine(const int* limits_global, int count, 
	DestIt dest_global, const T* carryIn_global, T identity, Op op, 
	CudaContext& context) {

	const int NT = 128;
  int numBlocks= MGPU_DIV_UP(count, NT);

	// Fix-up the segment outputs between the original tiles.
	MGPU_MEM(T) carryOutDevice = context.Malloc<T>(numBlocks);
	KernelSegReduceSpine1<NT><<<numBlocks, NT, 0, context.Stream()>>>(
		limits_global, count, dest_global, carryIn_global, identity, op,
		carryOutDevice->get());
	MGPU_SYNC_CHECK("KernelSegReduceSpine1");

	// Loop over the segments that span the tiles of 
	// KernelSegReduceSpine1 and fix those.
	if(numBlocks > 1) {
		KernelSegReduceSpine2<NT><<<1, NT, 0, context.Stream()>>>(
			limits_global, numBlocks, count, NT, dest_global,
			carryOutDevice->get(), identity, op);
		MGPU_SYNC_CHECK("KernelSegReduceSpine2");
	}
}

// For SpMM
template<int TB, int NT, typename T, typename Op, typename DestIt>
MGPU_HOST void SegReduceSpinePrealloc(const int* limits_global, int count, 
	DestIt dest_global, const T* carryIn_global, T* carryOut_global, T identity, 
  Op op, int B_ncols, CudaContext& context) {

  dim3 nb, nt;
  nt.x = NT;
  nt.y = 1;
  nt.z = 1;
  nb.x = MGPU_DIV_UP(32*count, NT);
  nb.y = 1;
  nb.z = MGPU_DIV_UP(B_ncols,32);

	// Fix-up the segment outputs between the original tiles.
	KernelSegReduceSpine1Prealloc<TB,NT><<<nb, nt, 0, context.Stream()>>>(
		limits_global, count, dest_global, carryIn_global, identity, op,
		carryOut_global, B_ncols);
	MGPU_SYNC_CHECK("KernelSegReduceSpine1");

	// Loop over the segments that span the tiles of 
	// KernelSegReduceSpine1 and fix those.
  dim3 nb2, nt2;
  nt2.x = NT;
  nt2.y = 1;
  nt2.z = 1;
  nb2.x = 1;
  nb2.y = 1;
  nb2.z = nb.z;
	if(nb.x > 1) {
		KernelSegReduceSpine2Prealloc<TB,NT><<<nb2, nt2, 0, context.Stream()>>>(
			limits_global, nb.x, count, dest_global, carryOut_global, identity, 
      B_ncols, op);
		MGPU_SYNC_CHECK("KernelSegReduceSpine2");
	}
}
////////////////////////////////////////////////////////////////////////////////
// Common LaunchBox structure for segmented reductions.

template<int NT_, int VT_, int OCC_, bool HalfCapacity_, bool LdgTranspose_>
struct SegReduceTuning {
	enum { 
		NT = NT_,
		VT = VT_, 
		OCC = OCC_,
		HalfCapacity = HalfCapacity_,
		LdgTranspose = LdgTranspose_
	};
};

} // namespace mgpu
