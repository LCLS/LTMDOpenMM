#ifndef OPENMM_LTMD_ANALYSIS_H_
#define OPENMM_LTMD_ANALYSIS_H_

#include <map>
#include <utility>
#include <vector>

#include "OpenMM.h"
#include "LTMD/Parameters.h"
#include "LTMD/Matrix.h"

namespace OpenMM {
	namespace LTMD {
		struct Block {
			unsigned int StartAtom, EndAtom;
			Matrix Data;

			Block() : StartAtom( 0 ), EndAtom( 0 ), Data() {}
			Block( size_t start, size_t end ) : StartAtom( start ), EndAtom( end ), Data( end - start + 1, end - start + 1 ) {}
		};

		typedef std::vector<double> EigenvalueArray;
		typedef std::pair<double, int> EigenvalueColumn;

		class OPENMM_EXPORT Analysis {
			public:
				Analysis() : mParticleCount( 0 ), mLargestBlockSize( -1 ) {
					mInitialized = false;
					blockContext = NULL;
				}
				~Analysis() {
					if( blockContext ) {
						delete blockContext;
					}
				}
				void computeEigenvectorsFull( ContextImpl &contextImpl, const Parameters &params );
				const std::vector<std::vector<Vec3> > &getEigenvectors() const {
					return eigenvectors;
				}
				unsigned int blockNumber( int );
				bool inSameBlock( int, int, int, int );

				const Matrix CalculateU( const Matrix &E, const Matrix &Q ) const;
				static std::vector<EigenvalueColumn> SortEigenvalues( const EigenvalueArray &values );

				void Initialize( ContextImpl &context, const Parameters &ltmd );
				void DiagonalizeBlocks( const Matrix &hessian, const std::vector<Vec3> &positions, std::vector<double> &eval, Matrix &evec );
				static void DiagonalizeBlock( const Block &block, const std::vector<Vec3> &positions, const std::vector<double> &Mass, std::vector<double> &eval, Matrix &evec );
				static void GeometricDOF( const int size, const int start, const int end, const std::vector<Vec3> &positions, const std::vector<double> &Mass, std::vector<double> &eval, Matrix &evec );
			private:
				unsigned int mParticleCount;
				std::vector<double> mParticleMass;

				int mLargestBlockSize;
				bool mInitialized;
				std::vector<std::pair<int, int> > bonds;
				std::vector<std::vector<int> > particleBonds;
				std::vector<std::vector<double> > projection;
				std::vector<std::vector<Vec3> > eigenvectors;
				Context *blockContext;
				std::vector<int> blocks;
		};
	}
}

#endif // OPENMM_LTMD_ANALYSIS_H_
