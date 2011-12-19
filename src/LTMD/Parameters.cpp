#include "LTMD/Parameters.h"

namespace OpenMM {
	namespace LTMD {
		Parameters::Parameters() {
			blockDelta = 1e-4; //NM
                        sDelta = 1e-4; // NM
			
			rediagFreq = 1000;
			minLimit = 0.41804;
			ShouldForceRediagOnMinFail = false;
			BlockDiagonalizePlatform = Preference::OpenCL;
		}
	}
}
