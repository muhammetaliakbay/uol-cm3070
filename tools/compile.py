import itertools
import os
import cv2
import h5py
import numpy as np


def scan(dataset_dir: str):
    for volume in itertools.count(1):
        volume_tag = f"{volume:03}"

        scenes = []

        for scene in itertools.count(1):
            scene_tag = f"{scene:03}"

            scene_dir = f"{dataset_dir}/ai_{volume_tag}_{scene_tag}/images"

            trajectories = []

            for trajectory in itertools.count(0):
                trajectory_tag = f"{trajectory:02}"

                preview_dir = \
                    f"{scene_dir}/scene_cam_{trajectory_tag}_final_preview"
                geometry_dir = \
                    f"{scene_dir}/scene_cam_{trajectory_tag}_geometry_hdf5"

                frames = []

                for frame in itertools.count(0):
                    frame_tag = f"{frame:04}"

                    color_file = f"{preview_dir}/frame.{frame_tag}.color.jpg"
                    depth_file = \
                        f"{geometry_dir}/frame.{frame_tag}.depth_meters.hdf5"

                    if not (
                        os.path.exists(color_file)
                        and os.path.exists(depth_file)
                    ):
                        break

                    frames.append((color_file, depth_file))

                if not frames:
                    break

                trajectories.append(frames)

            if not trajectories:
                break

            scenes.append(trajectories)

        if not scenes:
            break

        yield from (
            trajectory
            for scene in scenes
            for trajectory in scene
        )


if __name__ == "__main__":
    width = 1024
    height = 768
    downscale = 4
    far = 5.0
    outer = 6.0
    for trajectory, frames in enumerate(scan("source_dataset")):
        print(f"Compiling trajectory {trajectory:03}...")
        gray_tensor = np.empty(
            (len(frames), height//downscale, width//downscale),
            dtype=np.uint8,
        )
        depth_tensor = np.empty(
            (len(frames), height//downscale, width//downscale),
            dtype=np.uint8,
        )

        for frame, (color_file, depth_file) in enumerate(frames):
            color_matrix = cv2.imread(color_file, cv2.IMREAD_COLOR)
            color_matrix = cv2.resize(
                color_matrix,
                (width//downscale, height//downscale),
                interpolation=cv2.INTER_AREA
            )
            gray_tensor[frame] = color_matrix.mean(axis=-1)

            with h5py.File(depth_file, "r") as depth_matrix_file:
                depth_matrix = depth_matrix_file["dataset"][:]
                depth_matrix[np.isnan(depth_matrix)] = np.Inf
                org_depth_matrix = depth_matrix
                depth_matrix = np.clip(
                    depth_matrix, 0.0, far
                )
                depth_matrix[org_depth_matrix > far] = outer
                depth_matrix = (255 * depth_matrix / outer).round()
                depth_tensor[frame] = cv2.resize(
                    depth_matrix.astype(np.uint8),
                    (width//downscale, height//downscale),
                    interpolation=cv2.INTER_AREA
                )

        with h5py.File(f"dataset/{trajectory:03}.hdf5", "w") as trajectory_file:
            trajectory_file.create_dataset("gray", data=gray_tensor)
            trajectory_file.create_dataset("depth", data=depth_tensor)
