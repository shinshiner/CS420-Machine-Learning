from GMM import GMM_EM
from VBEM import VBEM
from pyCompatible import *

pyPatch()

if __name__ == '__main__':
	gmm = VBEM(n_components = 10)
	gmm.select()
	gmm.show()