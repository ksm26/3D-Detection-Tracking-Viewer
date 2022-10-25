from viewer.viewer import Viewer
import numpy as np
from dataset.nuscenes_dataset import nuscenesTrackingDataset
import re
from config.config import cfg, cfg_from_yaml_file
from nuscenes import NuScenes

def nuscenes_viewer():
    root="/media/storage/kitti/kitti_training"
    label_path = f"/home/Abhishek/Deep_networks/3D-Multi-Object-Tracker/evaluation/results/nuscenes_mini/data/0001.txt"
    scene_no = int(re.findall(r'\d+', label_path)[-1])

    yaml_file = "./config/online/pvrcnn_mot_nuscenes.yaml"
    config = cfg_from_yaml_file(yaml_file,cfg)
    nusc = NuScenes(version="v1.0-mini", verbose=True, dataroot="/media/storage/nuscenes/v1.0-mini")
    dataset = nuscenesTrackingDataset(config, nusc, type=[config.tracking_type],label_path=label_path)
    # dataset = KittiTrackingDataset(root,seq_id=sequence_id,label_path=label_path)

    vi = Viewer(box_type="OpenPCDet")

    seq_list = config.tracking_seqs  # the tracking sequences
    print("tracking seqs: ", seq_list)

    current_scene = nusc.scene[scene_no]
    first_token = current_scene['first_sample_token']
    last_token = current_scene['last_sample_token']
    nbr_samples = current_scene['nbr_samples']
    current_token = first_token

    for i in range(nbr_samples):
        current_sample = nusc.get('sample', current_token)
        nuscenes_data = dataset[i,current_scene,current_token,current_sample]
        # pose_matrix = Quaternion(pose['rotation'].inverse).rotation_matrix
        next_token = current_sample['next']
        current_token = next_token


        if nuscenes_data['labels'] is not None:
            mask = (nuscenes_data['label_names']=="car")
            labels = nuscenes_data['labels'][mask]
            label_names = nuscenes_data['label_names'][mask]
            vi.add_3D_boxes(labels, ids=labels[:, -1].astype(int), box_info=label_names,caption_size=(0.09,0.09))
            vi.add_3D_cars(labels, ids=labels[:, -1].astype(int), mesh_alpha=1)
        # vi.add_points(nuscenes_data['points'].points[:3,:].T)

        vi.add_image(nuscenes_data['image'])
        lidar_2_cam =  np.dot(nuscenes_data['Ego2Cam'],np.linalg.inv(nuscenes_data['lidar_sensor_pose']))
        vi.set_extrinsic_mat(lidar_2_cam )
        viewpad = np.eye(4)
        viewpad[:3, :3] = nuscenes_data['camera_intrinsic']
        vi.set_intrinsic_mat(viewpad )

        vi.show_2D()
        # vi.save_2D()

        # vi.show_3D()


if __name__ == '__main__':
    nuscenes_viewer()
