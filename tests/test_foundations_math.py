import unittest
import numpy as np
from labs.foundations.linear_regression import loss_and_grad, train

class FoundationMathTests(unittest.TestCase):
    def test_hand_calculated_gradient(self):
        loss,dw,db=loss_and_grad([0,1,2],[1,3,5],0,0)
        self.assertAlmostEqual(loss,35/3)
        self.assertAlmostEqual(dw,-26/3)
        self.assertAlmostEqual(db,-6)

    def test_gradients_match_finite_difference(self):
        x,y=np.array([0,1,2]),np.array([1,3,5])
        w,b,h=.7,.4,1e-5
        _,dw,db=loss_and_grad(x,y,w,b)
        finite_w=(loss_and_grad(x,y,w+h,b)[0]-loss_and_grad(x,y,w-h,b)[0])/(2*h)
        finite_b=(loss_and_grad(x,y,w,b+h)[0]-loss_and_grad(x,y,w,b-h)[0])/(2*h)
        self.assertAlmostEqual(dw,finite_w,places=7)
        self.assertAlmostEqual(db,finite_b,places=7)

    def test_training_converges_on_known_line(self):
        w,b,history=train([0,1,2],[1,3,5])
        self.assertLess(history[-1],1e-8)
        self.assertAlmostEqual(w,2,places=4)
        self.assertAlmostEqual(b,1,places=4)
        self.assertTrue(all(a>=b for a,b in zip(history,history[1:])))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): loss_and_grad([],[],0,0)
        with self.assertRaises(ValueError): loss_and_grad([1],[1,2],0,0)
        with self.assertRaises(ValueError): train([1],[2],rate=float('nan'))

if __name__=='__main__': unittest.main()
