import pytest
import numpy as np
pytest.importorskip("gudhi")
from kuramoto_sivashinsky import kuramoto_sivashinsky_2d

def test_small_simulation():
    np.random.seed(42)
    x,y,u=kuramoto_sivashinsky_2d(16,16,8,8,.01,.02,1.25)
    assert x.shape==y.shape==u.shape==(8,8)
    assert np.isfinite(u).all()
