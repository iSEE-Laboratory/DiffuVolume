from .kitti_dataset import KITTIDataset
from .kitti_dataset_1215 import KITTIDataset1215
from .sceneflow_dataset import SceneFlowDatset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "kitti1215": KITTIDataset1215
}
