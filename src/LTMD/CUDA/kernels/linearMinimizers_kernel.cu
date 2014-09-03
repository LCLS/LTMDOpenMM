typedef float Real;

extern "C" __global__ void kNMLLinearMinimize1_kernel( int numAtoms, int paddedNumAtoms, int numModes, float4 *velm, long long *force, float4 *modes, float *modeWeights ) {
	extern __shared__ float dotBuffer[];
	for( int mode = blockIdx.x; mode < numModes; mode += gridDim.x ) {
		// Compute the projection of the mass weighted force onto one normal mode vector.
		Real dot = 0.0f;
		for( int atom = threadIdx.x; atom < numAtoms; atom += blockDim.x ) {
			const Real scale = sqrt( velm[atom].w );
			const int modePos = mode * numAtoms + atom;

			float fx = (float)force[atom] / (float)0x100000000;
			float fy = (float)force[atom+1*paddedNumAtoms] / (float)0x100000000;
			float fz = (float)force[atom+2*paddedNumAtoms] / (float)0x100000000;
			float4 m = modes[modePos];

			dot += scale * ( fx * m.x + fy * m.y + fz * m.z );
		}
		dotBuffer[threadIdx.x] = dot;

		__syncthreads();
		if( threadIdx.x == 0 ) {
			Real sum = 0;
			for( int i = 0; i < blockDim.x; i++ ) {
				sum += dotBuffer[i];
			}
			modeWeights[mode] = sum;
		}
	}
}

extern "C" __global__ void kNMLLinearMinimize2_kernel( int numAtoms, int paddedNumAtoms, int numModes, float invMaxEigen, float4 *posq, float4 *posqP, float4 *velm,
						 long long *force, float4 *modes, float *modeWeights ) {
	// Load the weights into shared memory.
	extern __shared__ float weightBuffer[];
	for( int mode = threadIdx.x; mode < numModes; mode += blockDim.x ) {
		weightBuffer[mode] = modeWeights[mode];
	}
	__syncthreads();

	// Compute the projected forces and update the atom positions.
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const Real invMass = velm[atom].w, sqrtInvMass = sqrt( invMass ), factor = invMass * invMaxEigen;

		float3 f = make_float3( 0.0f, 0.0f, 0.0f );
		for( int mode = 0; mode < numModes; mode++ ) {
			float4 m = modes[mode * numAtoms + atom];
			float weight = weightBuffer[mode];
			f.x += m.x * weight;
			f.y += m.y * weight;
			f.z += m.z * weight;
		}

		f.x *= sqrtInvMass;
		f.y *= sqrtInvMass;
		f.z *= sqrtInvMass;
		posqP[atom] = make_float4( (float)force[atom] / (float)0x100000000 - f.x, (float)force[atom+paddedNumAtoms]/(float) 0x100000000 - f.y, (float) force[atom+2*paddedNumAtoms]/(float) 0x100000000  - f.z, 0.0f );

		float4 pos = posq[atom];
		pos.x += factor * posqP[atom].x;
		pos.y += factor * posqP[atom].y;
		pos.z += factor * posqP[atom].z;
		posq[atom] = pos;
	}
}
