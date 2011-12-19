#ifndef OPENMM_LTMD_PARAMETER_H_
#define OPENMM_LTMD_PARAMETER_H_

#include <vector>
#include <string>

namespace OpenMM {
	namespace LTMD {
		namespace Preference{
			enum EPlatform { Reference, OpenCL, CUDA };
		}
		
		struct Force {
			Force( std::string n, int i ) : name( n ), index( i ) {}
			std::string name;
			int index;
		};

		struct Parameters {
			double blockDelta;
		        double sDelta;
			std::vector<int> residue_sizes;
			int res_per_block;
			int bdof;
			std::vector<Force> forces;
			int modes;
			int rediagFreq;
			double minLimit;
			
			bool ShouldForceRediagOnMinFail;
			Preference::EPlatform BlockDiagonalizePlatform;
			
			unsigned int MaximumRediagonalizations;
			unsigned int MaximumMinimizationCutoff;
			unsigned int MaximumMinimizationIterations;
			
			Parameters();
		};
	}
}

#endif // OPENMM_LTMD_PARAMETER_H_
