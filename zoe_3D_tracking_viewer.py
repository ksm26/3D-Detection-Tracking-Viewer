from viewer.viewer import Viewer
import numpy as np
import re 

def read_detection_label(detection_list):

    boxes = []
    names = []
    for line in detection_list:
        this_name = line[2]
        frame_id = int(line[0])
        ob_id = int(line[1])
        if this_name != "DontCare":
            line = np.array(line[7:14],np.float32).tolist()
            line.append(ob_id)
            boxes.append(line)
            names.append(this_name)

    return np.array(boxes),np.array(names)


def zoe_viewer(image,track_obj,lidar_2_cam,camera_intrinsic):
    vi = Viewer(box_type="OpenPCDet")
    labels, label_names = read_detection_label(track_obj)

    if labels is not None:
        mask = (label_names=="car")
        labels = labels[mask]
        label_names = label_names[mask]
        vi.add_3D_boxes(labels, ids=labels[:, -1].astype(int), box_info=label_names,caption_size=(0.09,0.09))
        # vi.add_3D_cars(labels, ids=labels[:, -1].astype(int), mesh_alpha=1)
        
    # vi.add_points(points[:,:3])

    vi.add_image(image)
    vi.set_extrinsic_mat(lidar_2_cam)
    vi.set_intrinsic_mat(camera_intrinsic[:3, :3])
    # print(f'Extrinsic = {lidar_2_cam}')
    # print(f'intrinsic = {camera_intrinsic}')

    vi.show_2D()

    # vi.show_3D()


if __name__ == '__main__':
    zoe_viewer()
