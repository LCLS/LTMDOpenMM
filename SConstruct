import os

vars = Variables()
vars.Add( BoolVariable( 'intel', '', 0 ) )
vars.Add( BoolVariable( 'cuda', 'Set to build cuda plugin', 0 ) )
vars.Add( BoolVariable( 'test', 'Set to enable tests', 0 ) )

env = Environment( variables = vars )
if env.get( 'intel', 0 ):
	env.Tool( 'intelc' )
	env['ENV']['PATH'] = os.environ.get( 'PATH', '' )
	env['ENV']['LD_LIBRARY_PATH'] = os.environ.get( 'LD_LIBRARY_PATH', '' )
	env['ENV']['INTEL_LICENSE_FILE'] = os.environ.get('INTEL_LICENSE_FILE', '')

def add_definition( definition ):
	env.AppendUnique( CPPDEFINES = [definition] )

def include_directories( path ):
	env.AppendUnique( CPPPATH = [path] )

def link_directories( path ):
	env.AppendUnique( LIBPATH = [path] )

include_directories( "include" )

if 'INCLUDE' in os.environ:
	for item in os.environ['INCLUDE'].split(':'):
		include_directories( item )

if 'INCLUDE_PATH' in os.environ:
	for item in os.environ['INCLUDE_PATH'].split(':'):
		include_directories( item )

if 'LIBRARY_PATH' in os.environ:
	for item in os.environ['LIBRARY_PATH'].split(':'):
		link_directories( item )

openmm_dir = os.environ.get( 'OPENMM_HOME' )
if openmm_dir:
	link_directories( openmm_dir + "/lib" )
	link_directories( openmm_dir + "/lib/plugins" )
	include_directories( openmm_dir + "/include" )

openmm_source = os.environ.get( "OPENMM_SOURCE" )
print openmm_source
if openmm_source:
	include_directories( openmm_source + "/libraries/jama/include" )

	include_directories( openmm_source + "/libraries/sfmt/include" )

	include_directories( openmm_source + "/platforms/reference/src" )
	include_directories( openmm_source + "/platforms/reference/include" )

	include_directories( openmm_source + "/platforms/cuda/src" )
	include_directories( openmm_source + "/platforms/cuda/include" )

link_directories( "." )

base_sources = Glob( 'src/LTMD/*.cpp' )
env.Library( 'OpenMMLTMD', base_sources )

conf = Configure( env )
if not conf.CheckLib('OpenMM'):
	print 'Unable to find OpenMM library'
	Exit( 1 )

cuda = env.get( 'cuda', 0 )
if cuda and not conf.CheckLib( 'OpenMMCuda'):
	print 'Unable to find OpenMM Cuda Library'
	Exit( 1 )

conf.Finish()

reference_sources = Glob( 'src/LTMD/Reference/*.cpp' )
env.SharedLibrary( 'LTMDReference', reference_sources, LIBS = ['OpenMM','OpenMMLTMD'] )

if cuda:
	for item in env['CPPPATH']:
		env.AppendUnique( NVCCINC = ['-I' + item] )

	cuda_sources = Glob( 'src/LTMD/CUDA/*.cpp' )
	cuda_kernels = Glob( 'src/LTMD/CUDA/kernels/*.cu' )
	env.Tool( 'cuda' )
	env.SharedLibrary( 'LTMDCuda', cuda_sources + cuda_kernels, LIBS = ['OpenMM','OpenMMCuda','OpenMMLTMD'] )