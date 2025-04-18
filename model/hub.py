from model.model import MultiInputResShift
from huggingface_hub import PyTorchModelHubMixin

class MultiInputResShiftHub(
    MultiInputResShift, 
    PyTorchModelHubMixin,
    repo_url="https://github.com/VicFonch/Multi-Input-Resshift-Diffusion-VFI",
    paper_url="https://arxiv.org/pdf/2504.05402",
    language="en",
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)