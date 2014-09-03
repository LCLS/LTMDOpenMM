extern "C" __global__ void kRejectMinimizationStep_kernel( int numAtoms, float4 *posq, float4 *oldPosq ) {
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		posq[atom] = oldPosq[atom];
	}
}

extern "C" __global__ void kAcceptMinimizationStep_kernel( int numAtoms, float4 *posq, float4 *oldPosq ) {
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		oldPosq[atom] = posq[atom];
	}
}
