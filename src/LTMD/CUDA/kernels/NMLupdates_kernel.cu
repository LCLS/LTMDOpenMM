extern "C" __global__ void kNMLUpdate1_kernel( int numAtoms, int paddedNumAtoms, float tau, float dt, float kT, float4 *posq, float4 *posqP, float4 *velm, long long *force, const float4 *__restrict__ random, unsigned int randomIndex ) {
	/* Update the velocity.*/
	const float vscale = exp( -dt / tau );
	const float fscale = ( 1.0f - vscale ) * tau;
	const float noisescale = sqrt( kT * ( 1 - vscale * vscale ) );

	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const float4 n = random[randomIndex + blockIdx.x * blockDim.x + threadIdx.x];
		const float4 randomNoise = make_float4( n.x * noisescale, n.y * noisescale, n.z * noisescale, n.w * noisescale );

		const float sqrtInvMass = sqrt( velm[atom].w );

		float4 v = velm[atom];
		float fx = ( float )force[atom] / ( float )0x100000000;
		float fy = ( float )force[atom + 1 * paddedNumAtoms] / ( float )0x100000000;
		float fz = ( float )force[atom + 2 * paddedNumAtoms] / ( float )0x100000000;

		v.x = ( vscale * v.x ) + ( fscale * fx * v.w ) + ( randomNoise.x * sqrtInvMass );
		v.y = ( vscale * v.y ) + ( fscale * fy * v.w ) + ( randomNoise.y * sqrtInvMass );
		v.z = ( vscale * v.z ) + ( fscale * fz * v.w ) + ( randomNoise.z * sqrtInvMass );

		velm[atom] = v;
	}
}

extern "C" __global__ void kNMLUpdate2_kernel( int numAtoms, int numModes, float4 *velm, float4 *modes, float *modeWeights ) {
	extern __shared__ float dotBuffer[];
	for( int mode = blockIdx.x; mode < numModes; mode += gridDim.x ) {
		/* Compute the projection of the mass weighted velocity onto one normal mode vector. */
		float dot = 0.0f;

		for( int atom = threadIdx.x; atom < numAtoms; atom += blockDim.x ) {
			const int modePos = mode * numAtoms + atom;
			const float scale = 1.0f / sqrt( velm[atom].w );

			float4 v = velm[atom];
			float4 m = modes[modePos];

			dot += scale * ( v.x * m.x + v.y * m.y + v.z * m.z );
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

extern "C" __global__ void kNMLUpdate3_kernel( int numAtoms, int numModes, float dt, float4 *posq, float4 *velm, float4 *modes, float *modeWeights, float4 *noiseVal ) {
	/* Load the weights into shared memory. */
	extern __shared__ float weightBuffer[];
	for( int mode = threadIdx.x; mode < numModes; mode += blockDim.x ) {
		weightBuffer[mode] = modeWeights[mode];
	}
	__syncthreads();

	/* Compute the projected velocities and update the atom positions. */
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const float invMass = velm[atom].w, scale = sqrt( invMass );

		float3 v = make_float3( 0.0f, 0.0f, 0.0f );
		for( int mode = 0; mode < numModes; mode++ ) {
			float4 m = modes[mode * numAtoms + atom];
			float weight = weightBuffer[mode];
			v.x += m.x * weight;
			v.y += m.y * weight;
			v.z += m.z * weight;
		}

		v.x *= scale;
		v.y *= scale;
		v.z *= scale;
		velm[atom] = make_float4( v.x, v.y, v.z, invMass );

		float4 pos = posq[atom];

		/* Add Step */
		pos.x += dt * v.x;
		pos.y += dt * v.y;
		pos.z += dt * v.z;

#ifdef FAST_NOISE
		/* Remove Noise */
		pos.x -= noiseVal[atom].x;
		pos.y -= noiseVal[atom].y;
		pos.z -= noiseVal[atom].z;
#endif

		posq[atom] = pos;
	}
}
