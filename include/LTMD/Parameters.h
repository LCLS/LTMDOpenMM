#ifndef OPENMM_LTMD_PARAMETER_H_
#define OPENMM_LTMD_PARAMETER_H_

#include <vector>
#include <string>

namespace OpenMM {
	namespace LTMD {
		struct Force {
			Force( std::string n, int i ) : name( n ), index( i ) {}
			std::string name;
			int index;
		};

		struct Parameters {
			double delta;
			std::vector<int> residue_sizes;
			int res_per_block;
			int bdof;
			std::vector<Force> forces;
			int modes;
		        int rediagFreq;
		        double minLimit;
		};
	}
}

#endif // OPENMM_LTMD_PARAMETER_H_
