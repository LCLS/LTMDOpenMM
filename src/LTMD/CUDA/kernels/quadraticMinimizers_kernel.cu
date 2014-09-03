extern "C" __global__ void kNMLQuadraticMinimize1_kernel( int numAtoms, int paddedNumAtoms, float4 *posqP, float4 *velm, long long *force, float *blockSlope ) {
	/* Compute the slope along the minimization direction. */
	extern __shared__ float slopeBuffer[];

	float slope = 0.0f;
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const float invMass = velm[atom].w;
		const float4 xp = posqP[atom];
		const float fx = ( float )force[atom] / ( float )0x100000000;
		const float fy = ( float )force[atom + 1 * paddedNumAtoms] / ( float )0x100000000;
		const float fz = ( float )force[atom + 2 * paddedNumAtoms] / ( float )0x100000000;

		slope -= invMass * ( xp.x * fx + xp.y * fy + xp.z * fz );
	}
	slopeBuffer[threadIdx.x] = slope;
	__syncthreads();
	if( threadIdx.x == 0 ) {
		for( int i = 1; i <  blockDim.x; i++ ) {
			slope += slopeBuffer[i];
		}
		blockSlope[blockIdx.x] = slope;
	}
}

extern "C" __global__ void kNMLQuadraticMinimize2_kernel( int numAtoms, float currentPE, float lastPE, float invMaxEigen, float4 *posq, float4 *posqP, float4 *velm, float *blockSlope, float *lambdaval ) {
	/* Load the block contributions into shared memory. */
	extern __shared__ float slopeBuffer[];
	for( int block = threadIdx.x; block < gridDim.x; block += blockDim.x ) {
		slopeBuffer[block] = blockSlope[block];
	}

	__syncthreads();

	/* Compute the scaling coefficient. */
	if( threadIdx.x == 0 ) {
		float slope = 0.0f;
		for( int i = 0; i < gridDim.x; i++ ) {
			slope += slopeBuffer[i];
		}
		float lambda = invMaxEigen;
		float oldLambda = lambda;
		float a = ( ( ( lastPE - currentPE ) / oldLambda + slope ) / oldLambda );

		if( a != 0.0f ) {
			const float b = slope - 2.0f * a * oldLambda;
			lambda = -b / ( 2.0f * a );
		} else {
			lambda = 0.5f * oldLambda;
		}

		if( lambda <= 0.0f ) {
			lambda = 0.5f * oldLambda;
		}

		slopeBuffer[0] = lambda - oldLambda;

		/* Store variables for retrival */
		lambdaval[0] = lambda;
	}

	__syncthreads();

	/* Remove previous position update (-oldLambda) and add new move (lambda). */
	const float dlambda = slopeBuffer[0];
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const float factor = velm[atom].w * dlambda;

		float4 pos = posq[atom];
		pos.x += factor * posqP[atom].x;
		pos.y += factor * posqP[atom].y;
		pos.z += factor * posqP[atom].z;
		posq[atom] = pos;
	}
}
