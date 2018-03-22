from GMM import GMM_EM
from VBEM import VBEM
from kmeans import kmeans
from pyCompatible import *

pyPatch()

def testGMM(aic=True):
	model = GMM_EM(n_components = 10)
	if aic:
		print '------------test GMM with aic selection------------'
		model.aic_select()
	else:
		print '------------test GMM with bic selection------------'
		model.bic_select()
	model.show()

def testVBEM():
	print '------------test VBEM------------'
	model = VBEM(n_components = 10)
	model.select()
	model.show()

def testKmeans():
	print '------------test kmeans------------'
	model = kmeans(n_clusters = 10)
	model.select()
	model.show()

if __name__ == '__main__':
	testKmeans()
	testVBEM()
	testGMM()
	testGMM(False)