import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

ntu_120_rgbd_dataset_pkl = "./ntu120_3danno.pkl"
ntu_120_rgbd_ml_feat_sp = "./ntu120_ml_feats"

# ntu 120 joint configurations
# from chest mid upwards
upper_body_joints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24]
# starting from chest mid downwards
lower_body_joints = [0, 12, 13, 14, 15, 16, 17, 18, 19]
# remove thumbs, ankles, hip_right, hip_left, wrists
sparse_joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 17, 19, 20, 21, 23]
# wrist, thumb, and tip for left and right hand
hand_joints = [6, 7, 10, 11, 21, 22, 23, 24]
# ankle and foot tip for left and right feet
feet_joints = [14, 15, 18, 19]
all_joints = upper_body_joints+lower_body_joints
ntu_joint_configs = {"no_filter":None,
                    "upper_body":upper_body_joints,
                    "lower_body":lower_body_joints,
                    "hand":hand_joints,
                    "feet":feet_joints}
ntu_links = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)]
ntu_links = [(i-1, j-1) for (i, j) in ntu_links]


def read_pkl(fp):
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    return data


def filter_by_arr(kps, arr):
    """
    :param kps: keypoints of shape (num_people, num_frames, num_joints, num_channels)
    :param arr: list of vertices that we want to include
    :return: filtered keypoints
    """
    if arr is None:
        return kps.copy()
    kps_f = kps[:, :, arr, :]
    return kps_f


def get_all_angles(frame, upper_triangle=True):
    """
    get the angles between one vertex and all other vertices
    :param frame: a tensor of shape (num_vertices, num_coords)
    :return: a tensor of shape (num_vertices, num_vertices) where entry (i,j)
    is the angle between vertex i and j. If upper triangle is true, only
    get the entries above the main diagonal, flattened
    """
    num_vertices, num_coords = frame.shape
    norms = np.linalg.norm(frame, axis=-1)
    norms[norms == 0.0] = 0.00001
    norms = np.expand_dims(norms, axis=-1)
    norms = np.tile(norms, (1, num_coords))
    frame_n = frame/norms
    A = np.expand_dims(frame_n, axis=1)
    A = np.tile(A, (1, num_vertices, 1))
    B = np.tile(np.expand_dims(frame_n, axis=0), (num_vertices, 1, 1))
    C = np.multiply(A, B)
    dps = np.sum(C, axis=-1)
    angles = np.clip(dps, a_max=1.0, a_min=-1.0)
    angles = np.arccos(angles)
    angles[angles == np.nan] = 0.0
    angs = []
    if upper_triangle:
        for i in range(num_vertices):
            for j in range(i, num_vertices):
                if i != j:
                    angs.append(angles[i][j])
    angs = np.array(angs)
    return angs


def get_angle_motion(frames, order=1):
    num_frames, num_vertices, num_channels = frames.shape
    angles_per_frame = np.zeros((num_frames, num_vertices, num_vertices))
    for i, fn in enumerate(range(num_frames)):
        if i == 0:
            angles = get_all_angles(frames[fn])
            angles_shape = angles.shape[0]
            angles_per_frame = np.zeros((num_frames, angles_shape))
            angles_per_frame[i] = angles
        else:
            angles = get_all_angles(frames[fn])
            angles_per_frame[i] = angles
    for t in range(order):
        if t == 0:
            angles_motion = np.abs(angles_per_frame[1:, :] - angles_per_frame[:-1, :])
        else:
            angles_motion = np.abs(angles_motion[1:, :] - angles_motion[:-1, :])
    angles_motion = np.sum(angles_motion, axis=0)
    angles_motion = angles_motion/num_frames
    return angles_motion



def get_motion_features_dim(kps, dim, order=1):
    """
    :param kps: keypoints of shape (num_frames, num_joints, num_channels)
    :param order: the order of the motion
    :param dim: the number of channels, i.e. 2D or 3D representations
    :return: average sum of motion values across a dimension, summed over vertices,
    averaged over the number of frames
    """
    n_frames = kps.shape[0]
    kps_i = kps[:, :, dim]
    for t in range(order):
        if t == 0:
            motion_vec = np.abs(kps_i[1:, :] - kps_i[:-1, :])
        else:
            motion_vec = np.abs(motion_vec[1:, :] - motion_vec[:-1, :])
    motion = np.sum(motion_vec.reshape(-1))/n_frames
    return motion


def get_motion_features_vertices(kps, dim, order=1):
    """
    :param kps: keypoints of shape (num_frames, num_joints, num_channels)
    :param order: the order of the motion
    :param dim: the number of channels, i.e. 2D or 3D representations
    :return: an array of motion values across a dimension
    """
    n_frames = kps.shape[0]
    kps_i = kps[:, :, dim]
    for t in range(order):
        if t == 0:
            motion_vec = np.abs(kps_i[1:, :] - kps_i[:-1, :])
        else:
            motion_vec = np.abs(motion_vec[1:, :] - motion_vec[:-1, :])
    motion = np.sum(motion_vec, axis=0)/n_frames
    return motion


def get_link_vals(kps):
    """
    :param kps: keypoints of shape (num_people, num_frames, num_joints, num_channels)
    :return: the bone keypoints
    """
    links = np.zeros_like(kps)
    for (a, b) in ntu_links:
        links[:, :, a, :] = kps[:, :, a, :] - kps[:, :, b, :]
    return links


def center_data(kps, center_val=1):
    center_joint = np.expand_dims(np.expand_dims(np.expand_dims(kps[0, 0, center_val, :], axis=0), axis=0), axis=0)
    center_joint = np.tile(center_joint, reps=[kps.shape[0], kps.shape[1], kps.shape[2], 1])
    kps_ = kps-center_joint
    return kps_



def gen_kp_features_ntu(kps_):
    """
    :param kps: keypoints of shape (num_people, num_frames, num_joints, num_channels)
    :return: tensor of features of shape (num_features)
    """
    feats = []
    kps = center_data(kps_)
    num_people = kps.shape[0]
    for p_n in range(num_people):
        sub_feats = []
        for dim in [0, 1, 2]:
            sub_feats.append(get_motion_features_vertices(kps_[p_n], dim))
        sub_feats.append(get_angle_motion(kps_[p_n]))
        sub_feats = np.concatenate([sub_feats[i] for i in range(len(sub_feats))], axis=0)
        feats.append(sub_feats)
    return feats


def get_pca_feats(feats, n_components=30):
    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)
    pca = PCA(n_components=n_components)
    feats = pca.fit_transform(feats)
    return feats


def strip_zeros(str):
    while str[0] == "0":
        str = str[1:]
    return int(str)


def get_action_class(inner_fp):
    start_val = inner_fp.find("A")
    action_class_str = inner_fp[start_val+1:start_val+4]
    action_class = strip_zeros(action_class_str)
    action_class = action_class-1 # ntu classes are 1 indexed
    return action_class


def gen_features_ntu(save_fp, with_pca=True):
    data = read_pkl(ntu_120_rgbd_dataset_pkl)
    feat_d_1p = {}
    feat_d_2p = {}
    print(f"Computing features")
    for i, ann in enumerate(data["annotations"]):
        if i % 100 == 0 and i != 0:
            print(f"Computed features for {i} keypoint entries.")
        fn = ann["frame_dir"]
        kp = ann["keypoint"]
        num_people = kp.shape[0]
        feats = gen_kp_features_ntu(kp)
        if num_people == 1:
            feat_d_1p[fn] = feats
        else:
            feat_d_2p[fn] = feats

    feat_d = {}
    feat_d_1p_keys = list(feat_d_1p.keys())
    num_features = feat_d_1p[feat_d_1p_keys[0]][0].shape[0]
    feats_1p = np.zeros((len(feat_d_1p_keys), num_features))
    for i, key in enumerate(feat_d_1p_keys):
        feats_1p[i] = feat_d_1p[key][0]

    if with_pca:
        print(f"Computing PCA for 1 person actions.")
        feats_1p = get_pca_feats(feats_1p)
        for i, key in enumerate(feat_d_1p_keys):
            feat_d[key] = {"features": feats_1p[i], "num_people": 1, "class": get_action_class(key)}

    feat_d_2p_keys = list(feat_d_2p.keys())
    num_features = feat_d_2p[feat_d_2p_keys[0]][0].shape[0]
    feats_2p = np.zeros((len(feat_d_2p_keys), num_features*2))
    for i, key in enumerate(feat_d_2p_keys):
        feats_2p[i, 0:num_features] = feat_d_2p[key][0]
        feats_2p[i, num_features:] = feat_d_2p[key][1]

    if with_pca:
        # PCA for 2 person actions
        print(f"Computing PCA for 2 person actions.")
        feats_2p = get_pca_feats(feats_2p)
        for i, key in enumerate(feat_d_2p_keys):
            feat_d[key] = {"features": feats_2p[i], "num_people": 2, "class": get_action_class(key)}

    if with_pca:
        print("Writing the features .pkl file")
        save_fp = save_fp+"_with_pca"
        save_fp += ".pkl"
        with open(save_fp, "wb") as f:
            pickle.dump(feat_d, f)
    else:
        print("Writing the .npz file")
        save_fp += ".npz" # more features saved with npz
        np.savez(save_fp, one_p=feats_1p, two_p=feats_2p, one_p_keys=feat_d_1p_keys, two_p_keys=feat_d_2p_keys)


if __name__ == "__main__":
    gen_features_ntu(ntu_120_rgbd_ml_feat_sp, with_pca=True)
    gen_features_ntu(ntu_120_rgbd_ml_feat_sp, with_pca=False)
