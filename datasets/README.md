## ct2mr data

Numpy data or default PNG images can be used for training. Note that depending of the data loader, each format needs a proper data processing.

### Numpy data
MRI must be scaled between [-1;1], site-wise. Files must be named `trainB.npy` and `valB.npy`.
CT must be clipped between 0 and 2500. Files must be named `trainA.npy` and `valA.npy`.

See `create_ct2mr_np.py`

### PNG data
MRI must be scaled between [-1;1], site-wise. The trick is to rescale between positive integers since PNG does not allow floating point. `trainB` and `valB`.
CT must be clipped between 0 and 2500. Files must be in folder `trainA` and `valA`.

