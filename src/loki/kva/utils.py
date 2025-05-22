import torch
import h5py


def save_tensor_to_hdf5(filename, tensor, mode):
    ig_tensor = torch.stack(tensor).to(dtype=torch.float16)
    array = ig_tensor.numpy()
    with h5py.File(filename, mode) as f:
        if "dataset" not in f:
            m, n = array.shape
            dset = f.create_dataset(
                "dataset",
                shape=(0, m, n),
                maxshape=(None, m, n),
                chunks=True,
                compression="gzip",
            )
        else:
            dset = f["dataset"]
            if array.shape != dset.shape[1:]:
                raise ValueError("Array shape mismatch")

        # 扩展数据集并写入
        dset.resize(dset.shape[0] + 1, axis=0)
        dset[-1] = array
