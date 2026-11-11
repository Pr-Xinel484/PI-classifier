import pytest
import numpy as np
pytest.importorskip("gudhi")
from persistence_image import PD

def test_sublevel_image():
    vec,h0,h1,*_=PD(np.arange(12,dtype=float).reshape(3,4),type="sublevel_set").PI(w=.1,res=(8,6),sigma=.03)
    assert vec.shape == (96,)
    assert h0.shape == h1.shape == (8,6)
    assert np.isfinite(vec).all()

def test_point_cloud_image():
    vec,*_=PD(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])).PI(w=.1,res=(4,4),sigma=.03)
    assert vec.shape == (32,)
    assert np.isfinite(vec).all()
