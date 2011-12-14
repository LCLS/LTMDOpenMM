#include "CudaTest.h"

#include "OpenMM.h"
#include "openmm/serialization/XmlSerializer.h"
#include "SimTKUtilities/RealVec.h"
#include "SimTKUtilities/SimTKOpenMMRealType.h"

#include "LTMD/Integrator.h"
#include "LTMD/Parameters.h"

#include <string>
#include <vector>
#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION( LTMD::Cuda::Test );

namespace LTMD {
	namespace Cuda {
		using namespace OpenMM;
		
		void assert_equal_tol( double should, double is, double tol) {
			double scale = std::abs(should) > 1.0 ? std::abs(should) : 1.0; 
			double diff = should - is;
			
			double value = std::abs( diff ) / scale;
			
			bool isLessThan = value <= tol;
			
			std::stringstream stream;
			stream << "Value " << value << " should not be less than " << tol;

			CPPUNIT_ASSERT_MESSAGE( stream.str(), isLessThan );
		};

		void Test::Initialisation() {
			Platform::loadPluginsFromDirectory(
				Platform::getDefaultPluginsDirectory()
			);

			// Create a context, just to initialize all plugins
			System system;
			
			OpenMM::NonbondedForce* nonbond = new OpenMM::NonbondedForce(); 
			system.addForce(nonbond);
	
			std::vector<OpenMM::Vec3> initPosInNm(3);
			for (int a = 0; a < 3; ++a) {
				initPosInNm[a] = Vec3(0.5*a,0,0); // location, nm

				system.addParticle(39.95); // mass of Ar, grams per mole

				// charge, L-J sigma (nm), well depth (kJ)
				nonbond->addParticle(0.0, 0.3350, 0.996); // vdWRad(Ar)=.188 nm
			}
	
			VerletIntegrator integrator(0.01);
			Context context(system, integrator);

			// Was NormalModeLangevin plugin loaded?
			std::vector<std::string> kernelName;
			kernelName.push_back("IntegrateNMLStep");
			
			CPPUNIT_ASSERT_NO_THROW( Platform::findPlatform(kernelName) );
		}
		
		void Test::MinimizeIntegration() {
			Platform::loadPluginsFromDirectory(
				Platform::getDefaultPluginsDirectory()
			);
			
			// Load the system.
			std::ifstream fSystem("data/villin.xml");
			CPPUNIT_ASSERT( fSystem.good() );
			
			System* system = XmlSerializer::deserialize<System>(fSystem);
			fSystem.close();
			
			std::ifstream fMin("data/villin_minimize.txt");
			CPPUNIT_ASSERT( fMin.good() );
			
			int numParticles = system->getNumParticles();

			// Load the starting positions.
			std::vector<Vec3> positions(numParticles);
			for (int i = 0; i < numParticles; i++) {
				fMin >> positions[i][0];
				fMin >> positions[i][1];
				fMin >> positions[i][2];
			}
			fMin.close();
			
			std::cout << "Particles: " << numParticles << " " << positions.size() << std::endl;

			// Create the integrator and context, then minimize it.
			int numModes = 10;

			int res[] = {21, 11, 12, 15, 12, 20, 16, 6, 10, 16,
						 20, 7, 17, 14, 13, 3, 3, 5, 11, 10,
				 20, 10, 9, 5, 19, 14, 19, 10, 14, 16,
				 6, 12, 5, 12, 5, 8, 3, 6, 19, 16,
				 6, 16, 6, 15, 16, 6, 7, 19};
			OpenMM::LTMD::Parameters ltmd;
			ltmd.delta = 1e-9;
			ltmd.bdof = 12;
			ltmd.res_per_block = 1;
			ltmd.modes = 20;
			ltmd.rediagFreq = 1000;
			for (int i = 0; i < 49; i++)
			   ltmd.residue_sizes.push_back(res[i]);

			ltmd.forces.push_back(OpenMM::LTMD::Force("CenterOfMass", 0));
			ltmd.forces.push_back(OpenMM::LTMD::Force("Bond", 1));
			ltmd.forces.push_back(OpenMM::LTMD::Force("Angle", 2));
			ltmd.forces.push_back(OpenMM::LTMD::Force("Dihedral", 3));
			ltmd.forces.push_back(OpenMM::LTMD::Force("Improper", 4));
			ltmd.forces.push_back(OpenMM::LTMD::Force("Nonbonded", 5));


			OpenMM::LTMD::Integrator integ(300, 100.0, 0.05, &ltmd);
			integ.setMaxEigenvalue(5e3);
			Context context(*system, integ, Platform::getPlatformByName("Cuda"));
			context.setPositions(positions);
			double energy1 = context.getState(State::Energy).getPotentialEnergy();
			integ.minimize(50);
			
			// Verify that the energy decreased, and the slow modes were not modified during minimization.
			State state = context.getState(State::Positions | State::Energy);
			CPPUNIT_ASSERT(state.getPotentialEnergy() < energy1);
			
			std::vector<std::vector<Vec3> > modes = integ.getProjectionVectors();
			
			const std::vector<Vec3>& newPositions = state.getPositions();
			for (int i = 0; i < numModes; i++) {
				double oldValue = 0.0, newValue = 0.0;
				for (int j = 0; j < numParticles; j++) {
					double scale = sqrt(system->getParticleMass(j));
					oldValue += positions[j].dot(modes[i][j])*scale;
					newValue += newPositions[j].dot(modes[i][j])*scale;
				}
				assert_equal_tol(oldValue, newValue, 1e-3);
			}

			// Simulate the system and see if the temperature is correct.
			integ.step(1);
			integ.setFriction(10);
			double ke = 0.0;
			const int steps = 5000;
			for (int i = 0; i < steps; ++i) {
				State state = context.getState(State::Energy);
				ke += state.getKineticEnergy();
				integ.step(1);
			}
			ke /= steps;
			double expected = 0.5*numModes*BOLTZ*300;
			assert_equal_tol(expected, ke, 6/std::sqrt((double) steps));

			delete system;
		}
	}
}
