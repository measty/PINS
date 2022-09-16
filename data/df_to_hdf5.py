import h5py
from pathlib import Path
import numpy as np
from numpy.random import default_rng

# given dataframe of set of slides, makes a virtual hdf5
# dataset to access and index in an easy manner
def to_binary_class(lab):
    d = {"Epithelioid": 0, "Biphasic": 1, "sarcomatoid": 1}
    # d={'Epithelioid': 1, 'Biphasic': 1, 'sarcomatoid': 0}
    return d[lab]


def df_to_hdf5(df, vds_path, data_path, max_patches=1000, use_all=True, epoch=0):
    rng = default_rng()
    fnames = df["WSI-FolderName"]
    entry_key = "patches"  # where the data is inside of the source files.

    df_labels = df["labels"]
    vsources = []
    sizes, orig_sizes, step_sizes = [], [], []
    labels, slide_labels, h_channel = [], [], []
    offset, offsets = 0, []
    with h5py.File(vds_path, "w", libver="latest") as f:
        for i, fname in enumerate(fnames):
            with h5py.File(data_path.joinpath(fname + ".hdf5"), "r") as f2:
                dset = f2[entry_key]
                sh = dset.shape
                # n_patches=min(sh[0],max_patches)
                rough_step = max(1, sh[0] // max_patches)
                step_sizes.append(rough_step)
                if use_all:
                    offset = epoch % rough_step
                    offsets.append(offset)
                npatches = len(
                    range(offset, sh[0] - rough_step + offset + 1, rough_step)
                )
                orig_sizes.append(sh[0])
                sizes.append(npatches)
                vsources.append(h5py.VirtualSource(dset))
                labels.extend(np.repeat(to_binary_class(df_labels.iloc[i]), npatches))
                h_channel.extend(
                    f2["h_channel"][
                        offsets[i] : orig_sizes[i]
                        - step_sizes[i]
                        + offsets[i]
                        + 1 : step_sizes[i]
                    ]
                )
                slide_labels.append(to_binary_class(df_labels.iloc[i]))

        layout = h5py.VirtualLayout(
            shape=(sum(sizes), 3, 224, 224), dtype=h5py.h5t.STD_U8BE
        )
        ind = 0
        for i in range(len(fnames)):
            layout[ind : ind + sizes[i], :, :, :] = vsources[i][
                offsets[i] : orig_sizes[i]
                - step_sizes[i]
                + offsets[i]
                + 1 : step_sizes[i],
                :,
                :,
                :,
            ]
            ind += sizes[i]
        f.create_virtual_dataset(entry_key, layout, fillvalue=0)
        f.create_dataset("labels", dtype=h5py.h5t.STD_U8BE, data=labels)
        f.create_dataset("sizes", dtype="i4", data=sizes)
        f.create_dataset("slide_labs", dtype=h5py.h5t.STD_U8BE, data=slide_labels)
        f.create_dataset("h_channel", dtype=np.float32, data=h_channel)

    return labels
