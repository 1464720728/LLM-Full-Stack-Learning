import unittest
try:
    import numpy as np
except ImportError:
    np=None


@unittest.skipIf(np is None,'optional NumPy dependency not installed')
class TinyEncoderTests(unittest.TestCase):
    def setUp(self):
        from labs.knowledge.neural_encoder import TinyDualEncoder
        self.queries=['red apple','blue berry'];self.docs=['warm fruit','cool fruit']
        self.model=TinyDualEncoder(self.queries+self.docs,dim=3)
    def test_feature_mean(self):
        f=self.model.features(['red red apple'])
        self.assertAlmostEqual(float(f.sum()),1)
        self.assertAlmostEqual(f[0,self.model.vocab['red']],2/3)
    def test_finite_difference(self):
        _,gradient=self.model.loss_and_gradient(self.queries,self.docs)
        eps=1e-6
        for row,col in [(0,0),(2,1),(4,2)]:
            original=self.model.weights[row,col]
            self.model.weights[row,col]=original+eps
            plus=self.model.loss_and_gradient(self.queries,self.docs)[0]
            self.model.weights[row,col]=original-eps
            minus=self.model.loss_and_gradient(self.queries,self.docs)[0]
            self.model.weights[row,col]=original
            self.assertAlmostEqual(gradient[row,col],(plus-minus)/(2*eps),places=6)
    def test_training_reduces_loss(self):
        losses=self.model.train(self.queries,self.docs,steps=120)
        self.assertLess(losses[-1],losses[0]*0.1)
    def test_unknown_abstain(self):self.assertEqual(self.model.search('spaceship',self.docs),[])
    def test_permutation_invariance(self):
        self.assertTrue(np.allclose(self.model.encode(['red apple']),self.model.encode(['apple red'])))
    def test_dimension(self):self.assertEqual(self.model.encode(self.queries).shape,(2,3))
    def test_alignment_validation(self):
        with self.assertRaises(ValueError):self.model.loss_and_gradient(['x'],['x','y'])


if __name__=='__main__':unittest.main()
