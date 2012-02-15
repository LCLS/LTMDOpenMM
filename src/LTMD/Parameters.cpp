#include "LTMD/Parameters.h"

namespace OpenMM {
	namespace LTMD {
		Parameters::Parameters() {
			blockDelta = 1e-4; //NM
			sDelta = 1e-4; // NM
			
			rediagFreq = 1000;
			minLimit = 0.41804;
			ShouldForceRediagOnMinFail = false;
			ShouldForceRediagOnQuadratic = true;
			BlockDiagonalizePlatform = Preference::OpenCL;
			
			MaximumRediagonalizations = 5;
			MaximumMinimizationCutoff = 2;
			MaximumMinimizationIterations = 25;

			// 1/10 * ( 1 / MaxEigenvalue )
			MinimumLambdaValue = 2e-7;
			
			ShouldProtoMolDiagonalize = false;
		}
	}
}
