import os
import numpy as np
import trimesh
from PIL import Image
import pickle
from matplotlib import pyplot as plt

def draw2dseg(ax, annot, idx1, idx2, c="r", alpha=1, linewidth=1):
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]],
        [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha,
        linewidth=linewidth,
    )

def draw2djoints(ax, annots, links, alpha=1, linewidth=1, color=None):
    colors = ["r", "m", "b", "c", "g", "y", "b"]
    #colors = ["lime", "lime", "lime", "lime", "lime", "lime", "lime"]
    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            if color is not None:
                link_color = color[finger_idx]
            else:
                link_color = colors[finger_idx]
            draw2dseg(
                    ax,
                    annots,
                    finger_links[idx],
                    finger_links[idx + 1],
                    c=link_color,
                    alpha=alpha,
                    linewidth=linewidth,
            )

def visualize_joints_2d(
    ax,
    joints,
    joint_idxs=True,
    links=None,
    alpha=1,
    scatter=True,
    linewidth=2,
    color=None,
    joint_labels=None,
    axis_equal=False, #True,
):
    if links is None:
        links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    if scatter:
        ax.scatter(x, y, 1, "r")

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            if joint_labels is None:
                joint_label = str(row_idx)
            else:
                joint_label = str(joint_labels[row_idx])
            ax.annotate(joint_label, (row[0], row[1]))
    draw2djoints(
            ax, joints, links, alpha=alpha, linewidth=linewidth, color=color
    )
    if axis_equal:
        ax.axis("equal")



def draw2djoints_gt_black(ax, annots, links, alpha=1, linewidth=1, color=None):
    colors = ["k", "k", "k", "k", "k", "k", "k"]

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            if color is not None:
                link_color = color[finger_idx]
            else:
                link_color = colors[finger_idx]
            draw2dseg(
                    ax,
                    annots,
                    finger_links[idx],
                    finger_links[idx + 1],
                    c=link_color,
                    alpha=alpha,
                    linewidth=linewidth,
            )

def visualize_joints_2d_gt_black(
    ax,
    joints,
    joint_idxs=True,
    links=None,
    alpha=1,
    scatter=True,
    linewidth=2,
    color=None,
    joint_labels=None,
    axis_equal=False, #True,
):
    if links is None:
        links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    if scatter:
        ax.scatter(x, y, 1, "r")

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            if joint_labels is None:
                joint_label = str(row_idx)
            else:
                joint_label = str(joint_labels[row_idx])
            ax.annotate(joint_label, (row[0], row[1]))
    draw2djoints_gt_black(
            ax, joints, links, alpha=alpha, linewidth=linewidth, color=color
    )
    if axis_equal:
        ax.axis("equal")

def get_camintr(cam_calib):
    return np.array(cam_calib).astype(np.float32) # shape: (3, 3)

def get_obj_pose(cam_extr, affine_transform):
        # Get object pose (3,4) matrix in world coordinate frame
        transform = cam_extr @ affine_transform
        return np.array(transform).astype(np.float32)

def transform(verts, trans, convert_to_homogeneous=False):
    assert len(verts.shape) == 2, "Expected 2 dimensions for verts, got: {}.".format(len(verts.shape))
    assert len(trans.shape) == 2, "Expected 2 dimensions for trans, got: {}.".format(len(trans.shape))
    if convert_to_homogeneous:
        hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
    else:
        hom_verts = verts

    assert trans.shape[1] == hom_verts.shape[1], \
        "Incompatible shapes: verts.shape: {}, trans.shape: {}".format(verts.shape, trans.shape)

    trans_verts = np.dot(trans, hom_verts.transpose()).transpose()
    return trans_verts

def get_obj_corners3d(obj_path, cam_extr, affine_transform):
        #model = self.get_obj_verts_trans(obj_path, cam_extr, affine_transform)
        verts = trimesh.load(obj_path).vertices
        model = np.array(verts).astype(np.float32)
        x_values = model[:, 0]
        y_values = model[:, 1]
        z_values = model[:, 2]
        min_x, max_x = np.min(x_values), np.max(x_values)
        min_y, max_y = np.min(y_values), np.max(y_values)
        min_z, max_z = np.min(z_values), np.max(z_values)
        corners_3d = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        corners_3d = transform(corners_3d, get_obj_pose(cam_extr, affine_transform), convert_to_homogeneous=True)
        return np.array(corners_3d).astype(np.float32)


def get_objcorners2d(cam_calib, obj_path, cam_extr, affine_transform):
        corners_3d = get_obj_corners3d(obj_path, cam_extr, affine_transform)
        intr = get_camintr(cam_calib)
        corners_2d_hom = transform(corners_3d, intr)
        corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:]
        return np.array(corners_2d).astype(np.float32)

# Change this path
root = '/home/razvan/HUP-3D-model/data/combined_train'
evaluation = os.path.join(root)
all_files = os.path.join(root)
# Load object mesh
reorder_idx = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])

coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)


images_train = []
points2d_train = []
points3d_train = []

images_val = []
points2d_val = []
points3d_val = []

points_3d = np.load(os.path.join('/home/razvan/HUP-3D-model/data', 'points3d-test.npy'), allow_pickle=True)
images_test = np.load(os.path.join('/home/razvan/HUP-3D-model/data', 'images-test.npy'), allow_pickle=True)
# Train
rgb = os.path.join(all_files, 'rgb')
meta = os.path.join(all_files, 'meta')
for rgb_file in os.listdir(rgb):
    rgb_file = "img_rgb_29573.jpg"
    file_number = rgb_file.split('.')[0].split('_')[-1]
    meta_file = os.path.join(meta, 'img_meta_' + file_number+'.pkl')
    img_path = os.path.join(rgb, rgb_file)        
    metainfo = np.load(meta_file, allow_pickle=True)

    affine_transform = metainfo["affine_transform"]
    cam_calib = metainfo['cam_calib']
    obj_path = "/home/razvan/HUP-3D-model/data/models/voluson_painted_downscaled.ply"
    cam_extr = metainfo['cam_extr']

    cam_intr = get_camintr(cam_calib)
    obj_corners = get_obj_corners3d(obj_path, cam_extr, affine_transform)
    obj_corners2d = get_objcorners2d(cam_calib, obj_path, cam_extr, affine_transform)
    hand3d = metainfo['coords_3d'] #[reorder_idx]

    hand_object3d = np.concatenate([hand3d, obj_corners])
    hand_object3d = hand_object3d.dot(coordChangeMat.T)        
    hand_object_proj = cam_intr.dot(hand_object3d.transpose()).transpose()
    hand_object2d = (hand_object_proj / hand_object_proj[:, 2:])[:, :2]

    pil_img = Image.open(img_path)
    img = np.array(pil_img)
    height, width, _ = img.shape
    dpi = 200
    # Calculate the size of the figure in inches (width and height in inches)
    figsize = (width / dpi, height / dpi)

    # Create the figure with the calculated size
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.axis("off")
    visualize_joints_2d_gt_black(
            ax,
            obj_corners2d,
            # hand_object2d[21:,:],
            alpha=0.5,
            joint_idxs=False,
            links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]]
        )
    gt_handjoints2d = metainfo['coords_2d']
    visualize_joints_2d(ax, joints=gt_handjoints2d, joint_idxs=False, links=None, alpha=0.5, scatter=True, linewidth=2, color=None, joint_labels=None, axis_equal=False)
    ax.imshow(img)
    fig.tight_layout(pad=0)
    plt.show()
    images_train.append(img_path)
    points3d_train.append(hand_object3d)
    points2d_train.append(hand_object2d)



images_train = np.array(images_train)
points2d_train = np.array(points2d_train)
points3d_train = np.array(points3d_train)

images_val = np.array(images_val)
points2d_val = np.array(points2d_val)
points3d_val = np.array(points3d_val)

np.save('images-val.npy', images_val)
np.save('points2d-val.npy', points2d_val)
np.save('points3d-val.npy', points3d_val)

np.save('images-train.npy', images_train)
np.save('points2d-train.npy', points2d_train)
np.save('points3d-train.npy', points3d_train)


images_test = []
points2d_test = []
points3d_test = []
# Evaluation
for subject in os.listdir(os.path.join(evaluation)):
    s_path = os.path.join(evaluation, subject)
    rgb = os.path.join(s_path, 'rgb')
    meta = os.path.join(s_path, 'meta')
    for rgb_file in os.listdir(rgb):
        file_number = rgb_file.split('.')[0]
        meta_file = os.path.join(meta, file_number+'.pkl')
        img_path = os.path.join(rgb, rgb_file)
        data = np.load(meta_file, allow_pickle=True)
        cam_intr = data['camMat']
        hand3d = np.repeat(np.expand_dims(data['handJoints3D'], 0), 21, 0)
        hand3d = hand3d[reorder_idx]
        obj_corners = data['objCorners3D']
        hand_object3d = np.concatenate([hand3d, obj_corners])
        hand_object3d = hand_object3d.dot(coordChangeMat.T)
        hand_object_proj = cam_intr.dot(hand_object3d.transpose()).transpose()
        hand_object2d = (hand_object_proj / hand_object_proj[:, 2:])[:, :2]
        images_test.append(img_path)
        points3d_test.append(hand_object3d)
        points2d_test.append(hand_object2d)


images_test = np.array(images_test)
points2d_test = np.array(points2d_test)
points3d_test = np.array(points3d_test)

np.save('images-test.npy', images_test)
np.save('points2d-test.npy', points2d_test)
np.save('points3d-test.npy', points3d_test)

