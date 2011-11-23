#ifndef LTMDPARAMETERS_H
#define LTMDPARAMETERS_H

#include <vector>
#include <string>
#include <map>
using namespace std;

namespace OpenMM_LTMD {

struct LTMDForce {
   LTMDForce(string n, int i) : name(n), index(i) {}
   string name;
   int index;
};

struct LTMDParameters {
    double delta;
    vector<int> residue_sizes;
    int res_per_block;
    int bdof;
    vector<LTMDForce> forces;
    int modes;
};

}


#endif
