#include <iostream>
#include "LTMD/Reference/StepKernel.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/internal/ContextImpl.h"
#include "ReferenceCCMAAlgorithm.h"
#include "SimTKOpenMMUtilities.h"
#include <vector>

namespace OpenMM {
	namespace LTMD {
		namespace Reference {
			static std::vector<RealVec> &extractPositions( ContextImpl &context ) {
				ReferencePlatform::PlatformData *data = reinterpret_cast<ReferencePlatform::PlatformData *>( context.getPlatformData() );
				return *( ( std::vector<RealVec> * ) data->positions );
			}

			static std::vector<RealVec> &extractVelocities( ContextImpl &context ) {
				ReferencePlatform::PlatformData *data = reinterpret_cast<ReferencePlatform::PlatformData *>( context.getPlatformData() );
				return *( ( std::vector<RealVec> * ) data->velocities );
			}

			static std::vector<RealVec> &extractForces( ContextImpl &context ) {
				ReferencePlatform::PlatformData *data = reinterpret_cast<ReferencePlatform::PlatformData *>( context.getPlatformData() );
				return *( ( std::vector<RealVec> * ) data->forces );
			}

			StepKernel::~StepKernel() {

			}

			void StepKernel::initialize( const System &system, const Integrator &integrator ) {
				mParticles = system.getNumParticles();

				mMasses.resize( mParticles );
				mInverseMasses.resize( mParticles );
				for( unsigned int i = 0; i < mParticles; ++i ) {
					mMasses[i] = system.getParticleMass( i );
					mInverseMasses[i] = 1.0f / mMasses[i];
				}

				mXPrime.resize( mParticles );
				mPreviousPositions.resize( mParticles );

				SimTKOpenMMUtilities::setRandomNumberSeed( ( unsigned int ) integrator.getRandomNumberSeed() );
			}

			void StepKernel::Integrate( ContextImpl &context, const Integrator &integrator ) {
				// Calculate Constants
				const double deltaT = integrator.getStepSize();
				const double friction = integrator.getFriction();
				const double tau = friction == 0.0 ? 0.0 : 1.0 / friction;

				const double vscale = EXP( -deltaT / tau );
				const double fscale = ( 1 - vscale ) * tau;
				const double noisescale = std::sqrt( BOLTZ * integrator.getTemperature() * ( 1 - vscale * vscale ) );

				VectorArray &coordinates = extractPositions( context );
				VectorArray &velocities = extractVelocities( context );
				const VectorArray &forces = extractForces( context );

				// Update the velocity.
				for( unsigned int i = 0; i < mParticles; i++ ) {
					for( unsigned int j = 0; j < 3; j++ ) {
						const double velocity = vscale * velocities[i][j];
						const double force = fscale * forces[i][j];
						const double noise = noisescale * SimTKOpenMMUtilities::getNormallyDistributedRandomNumber();

						velocities[i][j] = velocity + force * mInverseMasses[i] + noise * std::sqrt( mInverseMasses[i] );
					}
				}

				// Project resulting velocities onto subspace
				Project( integrator, velocities, velocities, mMasses, mInverseMasses, false );

				// Update the positions.
				for( unsigned int i = 0; i < mParticles; i++ ) {
					coordinates[i][0] += deltaT * velocities[i][0];
					coordinates[i][1] += deltaT * velocities[i][1];
					coordinates[i][2] += deltaT * velocities[i][2];
				}
			}

			void StepKernel::UpdateTime( const Integrator &integrator ) {
				data.time += integrator.getStepSize();
				data.stepCount++;
			}

			void StepKernel::AcceptStep( ContextImpl &context ) {
				VectorArray &coordinates = extractPositions( context );

				for( unsigned int i = 0; i < mParticles; i++ ) {
					mPreviousPositions[i][0] = coordinates[i][0];
					mPreviousPositions[i][1] = coordinates[i][1];
					mPreviousPositions[i][2] = coordinates[i][2];
				}
				mMinimizerScale = 1.0;
			}

			void StepKernel::RejectStep( ContextImpl &context ) {
				VectorArray &coordinates = extractPositions( context );

				for( unsigned int i = 0; i < mParticles; i++ ) {
					coordinates[i][0] = mPreviousPositions[i][0];
					coordinates[i][1] = mPreviousPositions[i][1];
					coordinates[i][2] = mPreviousPositions[i][2];
				}
				mMinimizerScale *= 0.25;
			}

			void StepKernel::LinearMinimize( ContextImpl &context, const Integrator &integrator, const double energy ) {
				VectorArray &coordinates = extractPositions( context );
				const VectorArray &forces = extractForces( context );

				//save current PE in case quadratic required
				mPreviousEnergy = energy;

				//project forces into complement space, put in mXPrime
				Project( integrator, forces, mXPrime, mInverseMasses, mMasses, true );

				// Scale mXPrime if needed
				if( mMinimizerScale != 1.0 ) {
					for( unsigned int i = 0; i < mParticles; i++ ) {
						mXPrime[i][0] *= mMinimizerScale;
						mXPrime[i][1] *= mMinimizerScale;
						mXPrime[i][2] *= mMinimizerScale;
					}
				}

				//Add minimizer position update to atomCoordinates
				// with 'line search guess = 1/maxEig (the solution if the system was quadratic)
				for( unsigned int i = 0; i < mParticles; i++ ) {
					double factor = mInverseMasses[i] / integrator.getMaxEigenvalue();

					coordinates[i][0] += factor * mXPrime[i][0];
					coordinates[i][1] += factor * mXPrime[i][1];
					coordinates[i][2] += factor * mXPrime[i][2];
				}
			}

			double StepKernel::QuadraticMinimize( ContextImpl &context, const Integrator &integrator, const double energy ) {
				VectorArray &coordinates = extractPositions( context );
				const VectorArray &forces = extractForces( context );

				//Get quadratic 'line search' value
				double lambda = 1.0 / integrator.getMaxEigenvalue();
				const double oldLambda = lambda;

				//Solve quadratic for slope at new point
				//get slope dPE/d\lambda for quadratic, just equal to minus dot product of 'proposed position move' and forces (=-\nabla PE)
				double newSlope = 0.0;

				for( unsigned int i = 0; i < mParticles; i++ ) {
					for( unsigned int j = 0; j < 3; j++ ) {
						newSlope -= mXPrime[i][j] * forces[i][j] * mInverseMasses[i];
					}
				}

				//solve for minimum for quadratic fit using two PE vales and the slope with /lambda=0
				//for 'newSlope' use PE=a(\lambda_e-\lambda)^2+b(\lambda_e-\lambda)+c, \lambda_e is 1/maxEig.
				const double a = ( ( ( mPreviousEnergy - energy ) / oldLambda + newSlope ) / oldLambda );

				//calculate \lambda at minimum of quadratic fit
				if( a != 0.0 ) {
					const double b = newSlope - 2.0 * a * oldLambda;
					lambda = -b / ( 2.0 * a );
				} else {
					lambda = 0.5 * oldLambda;
				}

				//test if lambda negative, if so just use smaller lambda
				if( lambda <= 0.0 ) {
					lambda = 0.5 * oldLambda;
				}

				const double dlambda = lambda - oldLambda;

				//Remove previous position update (-oldLambda) and add new move (lambda)
				for( unsigned int i = 0; i < mParticles; i++ ) {
					const double factor = mInverseMasses[i] * dlambda;

					coordinates[i][0] += factor * mXPrime[i][0];
					coordinates[i][1] += factor * mXPrime[i][1];
					coordinates[i][2] += factor * mXPrime[i][2];
				}

				return lambda;
			}

			//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			// Find forces OR positions inside subspace (defined as the span of the 'eigenvectors' Q)
			// Take 'array' as input, 'outArray' as output (may be the same vector).
			//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			void CopyArray( const VectorArray &in, VectorArray &out ) {
				if( &in != &out ) {
					const unsigned int size = in.size();
					for( unsigned int i = 0; i < size; i++ ) {
						out[i][0] = in[i][0];
						out[i][1] = in[i][1];
						out[i][2] = in[i][2];
					}
				}
			}

			void ScaleArray( const DoubleArray &scale, VectorArray &out ) {
				const unsigned int size = out.size();
				for( unsigned int i = 0; i < size; i++ ) {
					const double weight = std::sqrt( scale[i] );

					out[i][0] *= weight;
					out[i][1] *= weight;
					out[i][2] *= weight;
				}
			}

			void StepKernel::Project( const Integrator &integrator, const VectorArray &in, VectorArray &out, const DoubleArray &scale, const DoubleArray &inverseScale, const bool compliment ) {
				unsigned int vectors = integrator.getNumProjectionVectors();

				if( mProjectionVectors.size() == 0 || integrator.getProjVecChanged() ) {
					if( mProjectionVectors.size() == 0 ) {
						const unsigned int size = vectors * mParticles * 3;

						mProjectionVectors.resize( size );
					}

					const std::vector<std::vector<OpenMM::Vec3> > &dProjectionVectors = integrator.getProjectionVectors();

					int index = 0;
					for( unsigned int i = 0; i < dProjectionVectors.size(); i++ ) {
						for( unsigned int j = 0; j < dProjectionVectors[i].size(); j++ ) {
							mProjectionVectors[index++] = dProjectionVectors[i][j][0];
							mProjectionVectors[index++] = dProjectionVectors[i][j][1];
							mProjectionVectors[index++] = dProjectionVectors[i][j][2];
						}
					}
				}

				CopyArray( in, out );
				ScaleArray( scale, out );

				const unsigned int _3N = in.size() * 3;

				//Project onto mode space by taking the matrix product of
				//the transpose of the eigenvectors Q with the array.
				//
				//c=Q^T*a', a' from last algorithm step
				//

				//If no Blas is available we need to manually find the product c=A*b
				//c_i=\sum_{j=1}^n A_{i,j} b_j

				//c=Q^T*a', a' from last algorithm step
				//Q is a linear array in column major format
				//so tmpC_i = \sum_{j=1}^n Q_{j,i} outArray_j
				//Q_{j,i}=_projectionVectors[j * numberOfAtoms * 3 + i]

				DoubleArray tmpC( vectors );
				for( unsigned int i = 0; i < vectors; i++ ) {

					tmpC[i] = 0.0;
					for( unsigned int j = 0; j < _3N; j++ ) {
						tmpC[i] += mProjectionVectors[j  + i * _3N] * out[j / 3][j % 3];
					}
				}

				//Now find projected force/positions a'' by matrix product with Eigenvectors Q
				//a''=Qc
				//so outArray_i  = \sum_{j=1}^n Q_{i,j} tmpC_i

				//find product
				for( unsigned int i = 0; i < _3N; i++ ) {

					//if sub-space do Q*c
					//else do a'-Q(Q^T a') = (I-QQ^T)a'
					const unsigned int ii = i / 3;
					const unsigned int jj = i % 3;
					if( !compliment ) {
						out[ii][jj] = 0.0;

						for( unsigned int j = 0; j < vectors; j++ ) {
							out[ii][jj] += mProjectionVectors[i + j * _3N] * tmpC[j];
						}
					} else {
						for( unsigned int j = 0; j < vectors; j++ ) {
							out[ii][jj] -= mProjectionVectors[i + j * _3N] * tmpC[j];
						}

					}

				}

				ScaleArray( inverseScale, out );
			}
		}
	}
}
