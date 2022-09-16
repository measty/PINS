# from numpy import vstack
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
import math, cv2
from tiatoolbox.tools import stainnorm
from numpy.random import default_rng
from PIL import ImageFilter
import h5py
import pickle

# extract patches from de-arrayed TMA cores, associate labels. Save patches for each core.

offsets = [3, 13, 25, 37]
slides = [4, 5, 6, 7]
base_path = Path(r"D:\All_cores")
foldernames, labels, slides = [], [], []

for slide_number in slides:
    psize = 224
    save_as = "hdf5"  #'patches' or 'hdf5'
    rng = default_rng()

    save_path = Path("E:\TCGA_Data\MESOv_hdf5")

    label_df = pd.read_csv(Path(r"D:\All_cores\core_labels.csv"))
    label_df.set_index("Core", inplace=True)
    slide_number = slide_number - 4

    offset = offsets[slide_number]

    images = list(base_path.glob("*.jpg"))

    norm = True
    um_per_pix = 0.44152

    if norm:
        # target_image_path='D:\TCGA_Data\Slides\stain_target.png'
        target_image_path = "D:\Meso\TMA_stain_target.tiff"
        target_image = cv2.imread(target_image_path)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        stain_normalizer = stainnorm.VahadaneNormaliser()
        stain_normalizer.fit(target_image)

    for img_path in images:
        print(f"processing {img_path.stem}..")
        stem_split = img_path.stem.split("-")
        row_ind = stem_split[0]
        column_ind = stem_split[1]
        label = label_df.loc[img_path.stem]["labels"]

        if label == "E":
            label = "Epithelioid"
        elif label == "B":
            label = "Biphasic"
        else:
            label = "sarcomatoid"

        with Image.open(img_path) as img:
            np_img = np.asarray(img)
            extractor = SlidingWindowPatchExtractor(
                np_img, (psize, psize), "otsu", pad_constant_values=255
            )
            coords = extractor.coord_list
            patches, patch_paths, stats = [], [], []
            for i, patch in enumerate(extractor):
                coord = coords[i, :]
                patch = Image.fromarray(patch)
                blur_patch = np.array(
                    patch.copy().filter(ImageFilter.GaussianBlur(radius=4))
                )
                pixel_mean = np.mean(blur_patch[:, :, 1:3], 2)
                white_pixels = np.mean(pixel_mean > 224)
                if white_pixels > 0.25:
                    continue
                if norm:
                    try:
                        normed_patch = stain_normalizer.transform(
                            np.asarray(patch).copy()
                        )
                    except:
                        print("norm failed, skipping patch")
                        continue
                    patches.append(np.transpose(normed_patch, (2, 0, 1)))
                else:
                    patches.append(patch)
                patch_paths.append(
                    save_path.joinpath(
                        f"{img_path.stem}_{coord[0]}_{coord[1]}_{label}.tiff"
                    )
                )

        npatches = len(patches)
        if npatches > 5:  # 5
            slides.append(label_df.loc[img_path.stem]["Slide"])
            if save_as == "hdf5":
                hdf_name = save_path.joinpath(f"{img_path.stem}_{label}.hdf5")
                foldernames.append(f"{img_path.stem}_{label}")
                labels.append(label)
                patches = np.stack(patches)

                with h5py.File(hdf_name, "w") as f:
                    f.create_dataset("patches", dtype=h5py.h5t.STD_U8BE, data=patches)
                    f.create_dataset("locs", dtype="i4", data=coords)
            else:
                for patch, spath in zip(patches, patch_paths):
                    patch.save(spath, "TIFF")
        else:
            print(f"too few usable patches found in core {img_path.stem}")

df = pd.DataFrame(foldernames, columns=["WSI-FolderName"])
df["labels"] = labels
# df['slide']=[slide_number]*len(df)
df["slide"] = slides

with open(save_path.joinpath(f"TMA_labels.pkl"), "wb") as output:
    pickler = pickle.Pickler(output, -1)
    pickler.dump(df)
    output.close()
