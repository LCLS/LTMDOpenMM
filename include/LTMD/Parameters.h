#ifndef OPENMM_LTMD_PARAMETER_H_
#define OPENMM_LTMD_PARAMETER_H_

#include <vector>
#include <string>

namespace OpenMM {
	namespace LTMD {
		namespace Preference {
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

			bool isAlwaysQuadratic;

			bool ShouldForceRediagOnMinFail;
			bool ShouldForceRediagOnQuadratic;
			bool ShouldForceRediagOnQuadraticLambda;
			Preference::EPlatform BlockDiagonalizePlatform;

			unsigned int MaximumMinimizationCutoff;
			unsigned int MaximumMinimizationIterations;

			double MinimumLambdaValue;

			int DeviceID;
			bool ShouldProtoMolDiagonalize;

			Parameters();
		};
	}
}

#endif // OPENMM_LTMD_PARAMETER_H_
