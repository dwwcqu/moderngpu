#pragma once

#define MGPU_NT  128 //64
#define MGPU_NTX 64
#define MGPU_VT  4  //4: Must be greater or equal to 4
#define MGPU_NV  MGPU_NTX*MGPU_VT
#define MGPU_TB  8  //8: Must be power of 4
#define MGPU_BC  64
