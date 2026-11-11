import pytest
import pandas as pd
import numpy as np
pytest.importorskip("gudhi")
from birdsong_delay_classifier import read_birdsong_metadata
from birdsong_mel_classifier import best_group_holdout_split

@pytest.mark.parametrize("col",["species","name"])
def test_metadata_column_alias(tmp_path,col):
    (tmp_path/"archive").mkdir()
    pd.DataFrame({col:["Song Sparrow"],"filename":["one.wav"]}).to_csv(tmp_path/"archive"/"bird_songs_metadata.csv",index=False)
    assert read_birdsong_metadata(tmp_path)["species"].tolist()==["Song Sparrow"]

def test_group_holdout():
    y=np.repeat(["a","b"],20)
    groups=np.array([f"group{i}" for i in range(40)])
    tr,te=best_group_holdout_split(y,groups,.2,42,50)
    assert not (set(groups[tr]) & set(groups[te]))
    assert set(y[tr])==set(y[te])=={"a","b"}
