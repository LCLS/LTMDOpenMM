#ifdef FAST_NOISE
extern "C" __global__ void kFastNoise1_kernel( int numAtoms, int paddedNumAtoms, int numModes, float kT, float4 *noiseVal, float4 *velm, float4 *modes, float *modeWeights, const float4 *__restrict__ random, unsigned int randomIndex, float maxEigenvalue, float stepSize ) {
	extern __shared__ float dotBuffer[];
	const float val = stepSize / 0.002;
	const float noisescale = sqrt( 2 * kT * 1.0f / maxEigenvalue );

	for( int mode = blockIdx.x; mode < numModes; mode += gridDim.x ) {
		float dot = 0.0f;
		unsigned int seed = 100;

		for( int atom = threadIdx.x; atom < numAtoms; atom += blockDim.x ) {
			const float4 n = random[randomIndex + blockIdx.x * blockDim.x + threadIdx.x];
			const float4 randomNoise = make_float4( n.x * noisescale, n.y * noisescale, n.z * noisescale, n.w * noisescale );

			noiseVal[atom] = randomNoise;

			float4 m = modes[mode * numAtoms + atom];
			dot += randomNoise.x * m.x + randomNoise.y * m.y + randomNoise.z * m.z;
		}

		dotBuffer[threadIdx.x] = dot;

		__syncthreads();
		if( threadIdx.x == 0 ) {
			float sum = 0;
			for( int i = 0; i < blockDim.x; i++ ) {
				sum += dotBuffer[i];
			}
			modeWeights[mode] = sum;
		}
	}
}

extern "C" __global__ void kFastNoise2_kernel( int numAtoms, int numModes, float4 *posq, float4 *noiseVal, float4 *velm, float4 *modes, float *modeWeights ) {
	/* Load the weights into shared memory.*/
	extern __shared__ float weightBuffer[];
	for( int mode = threadIdx.x; mode < numModes; mode += blockDim.x ) {
		weightBuffer[mode] = modeWeights[mode];
	}
	__syncthreads();

	/* Compute the projected forces and update the atom positions.*/
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const float invMass = velm[atom].w, sqrtInvMass = sqrt( invMass );

		float3 r = make_float3( 0.0f, 0.0f, 0.0f );
		for( int mode = 0; mode < numModes; mode++ ) {
			float4 m = modes[mode * numAtoms + atom];
			float weight = weightBuffer[mode];
			r.x += m.x * weight;
			r.y += m.y * weight;
			r.z += m.z * weight;
		}

		noiseVal[atom] = make_float4( noiseVal[atom].x - r.x, noiseVal[atom].y - r.y, noiseVal[atom].z - r.z, 0.0f );
		noiseVal[atom].x *= sqrtInvMass;
		noiseVal[atom].y *= sqrtInvMass;
		noiseVal[atom].z *= sqrtInvMass;

		float4 pos = posq[atom];
		pos.x += noiseVal[atom].x;
		pos.y += noiseVal[atom].y;
		pos.z += noiseVal[atom].z;
		posq[atom] = pos;
	}
}
#endif
