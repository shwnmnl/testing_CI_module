from main import extract_nifti_data, threshold_data, get_mean
import nibabel as nib
import numpy as np
import os


def test_extract_nifti_data(tmpdir):
    """Test the extract_nifti_data function."""
    data = np.ones((32,32,100), dtype=np.int16)
    img = nib.Nifti1Image(data, np.eye(4))
    path = os.path.join(tmpdir, "test_immg.nii.gz")
    nib.save(img, path)
    loaded_data = extract_nifti_data(path)
    assert np.array_equal(data, loaded_data), "Extraction did not work as expected"

def test_threshold_data():
    """Test the threshold_data function."""
    data = np.random.randn(4,4)
    threshold = 0.1
    thresholded_data = threshold_data(data, threshold)
    assert (thresholded_data > threshold).all(), "Thresholding did not work as expected"

def test_get_mean():   
    """Test the get_mean function."""
    data = np.ones((4,4))
    average = get_mean(data)
    assert average == 1, "Mean calculation did not work as expected"

