
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include "OpenMM.h"

using namespace std;
using namespace OpenMM;

class MyAtom;

class MyForceField {
public:
    // Atomic parameters
    virtual double getMassInAmu(const MyAtom&) const = 0;
    virtual double getChargeInE(const MyAtom&) const = 0;
    virtual double getVdwSigmaInNm(const MyAtom&) const = 0;
    virtual double getVdwEnergyInKJPerMole(const MyAtom&) const = 0;
    virtual double getBondLengthInNm(const MyAtom&, const MyAtom&) const = 0;
    virtual double getBondStiffnessInKJPerNmSquared(const MyAtom&, const MyAtom&) const = 0;
};

class MyAtom {
public:
    MyAtom(const string& atomName, const string& resName, int resNum, const Vec3& posInNm)
        : index(-1), name(atomName), residueName(resName), residueNumber(resNum), 
        defaultPositionInNm(posInNm), forceField(0)
    {}

    double getMassInAmu() const {
        return forceField->getMassInAmu(*this);
    }

    double getChargeInE() const {
        return forceField->getChargeInE(*this);
    }

    double getVdwSigmaInNm() const {
        return forceField->getVdwSigmaInNm(*this);
    }

    double getVdwEnergyInKJPerMole() const {
        return forceField->getVdwEnergyInKJPerMole(*this);
    }

    const Vec3& getDefaultPositionInNm() const {
        return defaultPositionInNm;
    }

    double getBondLengthInNm(const MyAtom& atom2) const {
        return forceField->getBondLengthInNm(*this, atom2);
    }

    double getBondStiffnessInKJPerNmSquared(const MyAtom& atom2) const {
        return forceField->getBondStiffnessInKJPerNmSquared(*this, atom2);
    }

    int index;
    string name;
    string residueName;
    int residueNumber;
    Vec3 defaultPositionInNm;
    const MyForceField* forceField;
};

class EthaneForceField : public MyForceField {
public:
    double getMassInAmu(const MyAtom& atom) const {
        if (bAtomIsC(atom)) return 12.001;
        if (bAtomIsH(atom)) return 1.008;
        throw runtime_error("Unrecognized atom");
    }

    double getChargeInE(const MyAtom& atom) const {
        if (bAtomIsC(atom)) return -0.1815;
        if (bAtomIsH(atom)) return  0.0605;
        throw runtime_error("Unrecognized atom");
    }

    double getVdwRadiusInNm(const MyAtom& atom) const {
        if (bAtomIsC(atom)) return 0.19080;
        if (bAtomIsH(atom)) return 0.14870;
        throw runtime_error("Unrecognized atom");
    }

    double getVdwSigmaInNm(const MyAtom& atom) const {
        return getVdwRadiusInNm(atom) * SigmaPerVdwRadius;
    }

    double getVdwEnergyInKJPerMole(const MyAtom& atom) const {
        if (bAtomIsC(atom)) return 0.4577;
        if (bAtomIsH(atom)) return 0.0657;
        throw runtime_error("Unrecognized atom");
    }

    double getBondLengthInNm(const MyAtom& atom1, const MyAtom& atom2) const {
        if (bAtomIsC(atom1) && bAtomIsC(atom2)) 
            return 0.1526;
        if ( (bAtomIsC(atom1) && bAtomIsH(atom2)) ||
             (bAtomIsH(atom1) && bAtomIsC(atom2)) )
            return 0.1090;
        throw runtime_error("Unrecognized bond");
    }

    double getBondStiffnessInKJPerNmSquared(const MyAtom& atom1, const MyAtom& atom2) const {
        if (bAtomIsC(atom1) && bAtomIsC(atom2)) 
            return 259408;
        if ( (bAtomIsC(atom1) && bAtomIsH(atom2)) ||
             (bAtomIsH(atom1) && bAtomIsC(atom2)) )
            return 284512;
        throw runtime_error("Unrecognized bond");
    }

    bool bAtomIsC(const MyAtom& atom) const {
        if (atom.residueName != "ETH") return false;
        if (atom.name.find(" C", 0) == 0) return true;
        return false;
    }

    bool bAtomIsH(const MyAtom& atom) const {
        if (atom.residueName != "ETH") return false;
        if (atom.name.find("H", 1) == 1) return true;
        return false;
    }
};

typedef pair<int, int> MyBond;

class MyMolecule {
public:
    typedef vector<MyAtom>::const_iterator const_atom_iterator;
    typedef vector< pair<int, int> >::const_iterator const_bond_iterator;

    MyMolecule(const MyForceField& forceField) 
        : coulomb14Scale(0.5), lennardJones14Scale(0.5), forceField(forceField)
    {}

    virtual ~MyMolecule() {}

    int add(const MyAtom& atom) {
        atoms.push_back(atom);
        atoms.back().forceField = &forceField;
        atoms.back().index = atoms.size() - 1;
        return atoms.back().index;
    }
    ostream& writePdb(const State& state, ostream& os) const;

    double coulomb14Scale;
    double lennardJones14Scale;
    vector<MyAtom> atoms;
    vector<MyBond> bonds;
    const MyForceField& forceField;
};

class EthaneMolecule : public MyMolecule {
public:
    EthaneMolecule(const Vec3& center = Vec3(0,0,0), int residueNumber = 1) 
        : MyMolecule(*(new EthaneForceField()))
    {
        int c1 = add(MyAtom(" C1 ", "ETH", residueNumber, Vec3(-0.07605,  0,   0)));
        int c2 = add(MyAtom(" C2 ", "ETH", residueNumber, Vec3(0.07605,  0,   0)));
        int h11 = add(MyAtom("1H1 ", "ETH", residueNumber, Vec3(-.1135, 0.103,  0)));
        int h21 = add(MyAtom("2H1 ", "ETH", residueNumber, Vec3(-.1135, -.051, 0.089)));
        int h31 = add(MyAtom("3H1 ", "ETH", residueNumber, Vec3(-.1135, -.051,-0.089)));
        int h12 = add(MyAtom("1H2 ", "ETH", residueNumber, Vec3(.1135, 0.103,  0)));
        int h22 = add(MyAtom("2H2 ", "ETH", residueNumber, Vec3(.1135, -0.051, 0.089)));
        int h32 = add(MyAtom("3H2 ", "ETH", residueNumber, Vec3(.1135, -0.051,-0.089)));

        bonds.push_back(make_pair(c1, c2));
        bonds.push_back(make_pair(c1, h11));
        bonds.push_back(make_pair(c1, h21));
        bonds.push_back(make_pair(c1, h31));
        bonds.push_back(make_pair(c2, h12));
        bonds.push_back(make_pair(c2, h22));
        bonds.push_back(make_pair(c2, h32));
    }

    ~EthaneMolecule() {
        delete &forceField;
    }
};

class ArgonForceField : public MyForceField {
    double getMassInAmu(const MyAtom& atom) const {
        return 39.948;
    }
    double getChargeInE(const MyAtom& atom) const {
        return 0.0;
    }
    double getVdwRadiusInNm(const MyAtom& atom) const {
        return 0.1880;
    }
    double getVdwSigmaInNm(const MyAtom& atom) const {
        return getVdwRadiusInNm(atom) * SigmaPerVdwRadius;
    }
    double getVdwEnergyInKJPerMole(const MyAtom& atom) const {
        return 0.9960;
    }
    double getBondLengthInNm(const MyAtom& atom1, const MyAtom& atom2) const {
        throw std::runtime_error("No bonded terms in argon");
    }
    double getBondStiffnessInKJPerNmSquared(const MyAtom& atom1, const MyAtom& atom2) const {
        throw std::runtime_error("No bonded terms in argon");
    }
};

class TwoArgons : public MyMolecule {
public:
    TwoArgons(const Vec3& center = Vec3(0,0,0)) 
        : MyMolecule(*(new ArgonForceField()))
    {
        add(MyAtom(" AR ", " AR", 1, Vec3(-0.250,  0,   0)));
        add(MyAtom(" AR ", " AR", 2, Vec3( 0.250,  0,   0)));
    }

    ~TwoArgons() {delete &forceField;}
};

class MyOpenMMSimulation {
public:
    MyOpenMMSimulation(const MyMolecule& mol) : 
        stepSizeInPs(0.002), bConstrainHBonds(false),
        molecule(mol), pdbModelCount(0)
    {
        Platform::loadPluginsFromDirectory
            (Platform::getDefaultPluginsDirectory());

        system = new System();

        // Establish initial atom positions and nonbonded parameters
        NonbondedForce* nonbond = new NonbondedForce();
        system->addForce(nonbond);
        std::vector<Vec3> positions;
        MyMolecule::const_atom_iterator a;
        for (a = molecule.atoms.begin(); a != molecule.atoms.end(); ++a) {
            system->addParticle(a->getMassInAmu());
            nonbond->addParticle(
                a->getChargeInE(),
                a->getVdwSigmaInNm(),
                a->getVdwEnergyInKJPerMole());
            positions.push_back(a->getDefaultPositionInNm());
        }

        HarmonicBondForce* bondStretch = new HarmonicBondForce();
        system->addForce(bondStretch);
        MyMolecule::const_bond_iterator b;
        for (b = molecule.bonds.begin(); b != molecule.bonds.end(); ++b)
        {
            const MyAtom& atom1 = molecule.atoms[b->first];
            const MyAtom& atom2 = molecule.atoms[b->second];
            bondStretch->addBond(atom1.index, atom2.index,
                atom1.getBondLengthInNm(atom2),
                atom1.getBondStiffnessInKJPerNmSquared(atom2));
        }

        // TODO - bonded forces
        HarmonicAngleForce* bondBend = new HarmonicAngleForce();
        system->addForce(bondBend);

        PeriodicTorsionForce* bondTorsion = new PeriodicTorsionForce();
        system->addForce(bondTorsion);

        integrator = new VerletIntegrator(stepSizeInPs);
        context = new Context(*system, *integrator);
        context->setPositions(positions);
    }

    ~MyOpenMMSimulation() {
        delete context; 
        delete integrator; 
        delete system;
    }

    void step(int numSteps) {
        integrator->step(numSteps);
    }

    ostream& writePdb(ostream& os = cout) {
        State state = context->getState(State::Positions);
        assert(state.getPositions().size() == molecule.atoms.size());
        int atomSerialNumber = 1;
        for (unsigned int a = 0; a < molecule.atoms.size(); ++a)
        {
            const MyAtom& atom = molecule.atoms[a];
            os << "ATOM  ";
            os << right << setw(5) << atomSerialNumber;
            os << " "; // blank at column 12
            os << setw(4) << atom.name;
            os << " "; // altloc
            os << setw(3) << atom.residueName;
            os << " "; // chain id
            os << right << setw(4) << atom.residueNumber;
            os << "    "; // blank columns 27-30

            // x,y,z coordinates
            Vec3 pos = state.getPositions()[a];
            for (int c = 0; c < 3; ++c) {
                double x = pos[c] * AngstromsPerNm;
                os << right << setw(8) << setiosflags(ios::fixed) << setprecision(3);
                os << x;
            }
            os << "  1.00"; // occupancy
            os << "  0.00"; // temperature factor
            os << "          "; // blank columns 67-76

            os << endl;
            ++atomSerialNumber;
        }
        return os;
    }

    // wrap coordinates in "MODEL/ENDMDL" for proper trajectory file
    ostream& writePdbModel(ostream& os = cout) {
        ++pdbModelCount;
        os << "MODEL ";
        os << right << setw(6) << pdbModelCount;
        os << endl;

        writePdb(os);

        os << "ENDMDL" << endl;

        return os;
    }

private:
    double stepSizeInPs;
    System*         system;
    Integrator*     integrator;
    Context*  context;
    const MyMolecule& molecule;
    bool   bConstrainHBonds;
    int pdbModelCount;
};


void test_ethane() 
{
    EthaneMolecule ethane;
    MyOpenMMSimulation sim(ethane);

    // ostream& os = cout;
    ofstream os("test.pdb");

    sim.writePdbModel(os);
    for (int s = 0; s < 10; ++s) {
        sim.step(1);
        sim.writePdbModel(os);
    }

    if (os != cout)
        os.close();
}

void test_argon()
{
    TwoArgons argons;
    MyOpenMMSimulation sim(argons);

    ofstream os("test.pdb");

    sim.writePdbModel(os);
    for (int s = 0; s < 100; ++s) {
        sim.step(100);
        sim.writePdbModel(os);
    }

    if (os != cout)
        os.close();
}

int main() {
    try {
        // test_ethane();
        test_argon();
        return 0; // success
    }
    catch(const exception& e) {
        printf("EXCEPTION: %s\n", e.what());
        return 1; // failure
    }
}
