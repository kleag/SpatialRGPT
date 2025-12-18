import cv2
import os
import matplotlib
import numpy as np
import sys
import torch
import torchvision.transforms as transforms
import trimesh
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# repo_id = "onnx-community/metric3d-vit-giant2"
repo_id = "onnx-community/metric3d-vit-large"
filename = "onnx/model.onnx"
# data_file = "onnx/model.onnx_data"
onnx_model_path = hf_hub_download(repo_id=repo_id, filename=filename)
# onnx_data_path = hf_hub_download(repo_id=repo_id, filename=data_file)

# --- Verification ---
# The two files should be in the same folder, which is the default cache location
model_dir = os.path.dirname(onnx_model_path)
print(f"Model Directory: {model_dir}", file=sys.stderr)
# Make sure model.onnx and model.onnx_data are both in this directory.

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
ort_session = ort.InferenceSession(
    onnx_model_path,
    providers=providers,
)

# Get input and output names
input_name = ort_session.get_inputs()[0].name
output_names = [output.name for output in ort_session.get_outputs()]


def get_depth_model(device):
    hub_dir = torch.hub.get_dir()
    repo_name = "yvanyin_metric3d_main"
    repo_path = os.path.join(hub_dir, repo_name)

    if os.path.exists(repo_path):
        # load from local cache
        print("Model metric3d_vit_giant2 cached.")
        depth_model = torch.hub.load(
            repo_path, "metric3d_vit_giant2", pretrained=True, source="local"
        )
    else:
        print("Model metric3d_vit_giant2 not cached. Loading from hub.")
        depth_model = torch.hub.load(
            "yvanyin/metric3d",
            "metric3d_vit_giant2",
            pretrain=True,
            force_reload=False,
            skip_validation=True,  # optional: skips online hash check
        )
    return depth_model.to(device)


def inference_depth(rgb_origin, intrinsic, depth_model):
    # Code from # https://github.com/YvanYin/Metric3D/blob/main/hubconf.py, assume rgb_origin is in RGB
    print(
        f"metric3d_v2 inference_depth initial intrinsic: {intrinsic}", file=sys.stderr
    )
    intrinsic = [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]

    #### ajust input size to fit pretrained model
    # keep ratio resize
    input_size = (616, 1064)  # for vit model
    # input_size = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(
        rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
    )
    # remember to scale intrinsic, hold depth
    intrinsic = [
        intrinsic[0] * scale,
        intrinsic[1] * scale,
        intrinsic[2] * scale,
        intrinsic[3] * scale,
    ]
    print(f"metric3d_v2 inference_depth scaled intrinsic: {intrinsic}", file=sys.stderr)

    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(
        rgb,
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=padding,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    print(f"metric3d_v2 inference_depth pad_info: {pad_info}", file=sys.stderr)

    #### normalize
    # mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    # std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    # rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    # rgb = torch.div((rgb - mean), std)
    # rgb = rgb[None, :, :, :].cuda()

    # dummy_input = torch.randn(1, 3, h, w).cuda() # H, W are the expected input_size H, W
    # with torch.no_grad():
    #     test_depth, _, _ = depth_model.inference({"input": dummy_input})
    # print(f"Dummy test output NaN check: {test_depth.isnan().any()}")

    # with torch.no_grad():
    #     pred_depth, confidence, output_dict = depth_model.inference({"input": rgb})
    # print(
    #     f"metric3d_v2 inference_depth initial pred_depth: {pred_depth}", file=sys.stderr
    # )

    # 1. Input Preparation (Using NumPy/CPU for normalization)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)[:, None, None]
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)[:, None, None]

    # Ensure 'rgb' is the PADDED NumPy array (H, W, C)
    rgb_np = rgb.transpose((2, 0, 1)).astype(np.float32)  # (C, H, W)
    normalized_input = np.divide((rgb_np - mean), std)
    final_onnx_input = normalized_input[np.newaxis, :, :, :]  # (1, C, H, W)

    # 2. ONNX Inference
    ort_inputs = {input_name: final_onnx_input}
    ort_outputs = ort_session.run(output_names, ort_inputs)

    # 3. Process Outputs
    # Assume the first output is depth, second is confidence (check model documentation!)
    pred_depth_np = ort_outputs[0]
    confidence_np = ort_outputs[1]

    # Convert NumPy arrays back to PyTorch tensors and move to CUDA
    # to continue with the rest of your original pipeline (e.g., Perspective Fields, Open3D).
    pred_depth = torch.from_numpy(pred_depth_np).cuda()
    confidence = torch.from_numpy(confidence_np).cuda()
    print(
        f"metric3d_v2 inference_depth onnx pred_depth: {pred_depth}, {confidence}",
        file=sys.stderr,
    )

    # un pad
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[
        pad_info[0] : pred_depth.shape[0] - pad_info[1],
        pad_info[2] : pred_depth.shape[1] - pad_info[3],
    ]

    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None, :, :], rgb_origin.shape[:2], mode="bilinear"
    ).squeeze()

    #### de-canonical transform
    canonical_to_real_scale = (
        intrinsic[0] / 1000.0
    )  # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 300)
    return pred_depth.detach().cpu().numpy()


def depth_to_mesh(points, depth, image_rgb):
    triangles = create_triangles(
        image_rgb.shape[0], image_rgb.shape[1], mask=~depth_edges_mask(depth)
    )
    mesh = trimesh.Trimesh(
        vertices=points.reshape(-1, 3),
        faces=triangles,
        vertex_colors=image_rgb.reshape(-1, 3),
    )
    # mesh_t.export(save_pcd_dir+f'/{filename}_t_mesh.obj')
    return mesh


def depth_edges_mask(depth):
    """Returns a mask of edges in the depth map.

    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx**2 + depth_dy**2)
    # Compute the edge mask.
    mask = depth_grad > 0.05
    return mask


def create_triangles(h, w, mask=None):
    """
    Reference: https://github.com/google-research/google-research/blob/e96197de06613f1b027d20328e06d69829fa5a89/infinite_nature/render_utils.py#L68
    Creates mesh triangle indices from a given pixel grid size.
        This function is not and need not be differentiable as triangle indices are
        fixed.
    Args:
    h: (int) denoting the height of the image.
    w: (int) denoting the width of the image.
    Returns:
    triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)
    """
    x, y = np.meshgrid(range(w - 1), range(h - 1))
    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1
    triangles = np.array([tl, bl, tr, br, tr, bl])
    triangles = np.transpose(triangles, (1, 2, 0)).reshape(((w - 1) * (h - 1) * 2, 3))
    if mask is not None:
        mask = mask.reshape(-1)
        triangles = triangles[mask[triangles].all(1)]
    return triangles


def get_intrinsics(H, W, fov):
    """Intrinsics for a pinhole camera model.

    Assume fov of 55 degrees and central principal point.
    """
    # fy = 0.5 * H / np.tan(0.5 * fov * np.pi / 180.0)
    # fx = 0.5 * W / np.tan(0.5 * fov * np.pi / 180.0)

    focal = H / 2 / np.tan(np.radians(fov) / 2)

    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])


def depth_to_points(depth, R=None, t=None, fov=None, intrinsic=None):
    if intrinsic is None:
        K = get_intrinsics(depth.shape[1], depth.shape[2], fov)
    else:
        K = intrinsic
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = np.eye(3)
    # M[0, 0] = -1.0
    # M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    # print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    # pts3D_2 = pts3D_1
    # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w

    # G converts from your coordinate to PyTorch3D's coordinate system
    G = np.eye(3)
    G[0, 0] = -1.0
    G[1, 1] = -1.0

    return pts3D_2[:, :, :, :3, 0][0] @ G.T

    # return (G[None, None, None, ...] @ pts3D_2)[:, :, :, :3, 0][0]

    # return pts3D_2[:, :, :, :3, 0][0]


def colorize_depth(
    value,
    vmin=None,
    vmax=None,
    cmap="inferno_r",
    invalid_val=-99,
    invalid_mask=None,
    background_color=(128, 128, 128, 255),
    gamma_corrected=False,
    value_transform=None,
):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask], 2) if vmin is None else vmin
    vmax = np.percentile(value[mask], 85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.0

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img
