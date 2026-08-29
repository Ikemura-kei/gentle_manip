"""Arm F: run THEIR GAP codebase end-to-end, swapping ONLY the RGB+depth branch for a point cloud.

The question this arm answers is ARCHITECTURE: C'/D'/E graft GAP's gradient rule onto OUR
point-cloud diffusion policy; F instead runs THEIR BCPolicy + THEIR FeatureExtractor +
THEIR DeterministicMLP head + THEIR WorkSpace trainer + THEIR CPD/LSTM phase, on OUR data.
If GAP helps there and not here, the architecture is the reason.

NOTHING in third_party/GAP is edited by this file. The only edit to their tree is the two-line
`lambda` keyword fix in gap/gap.py (their file does not parse as published). Everything below
is injected by monkeypatch, so their modules stay byte-identical.

FAITHFULNESS NOTES (each one is a deliberate decision, not an accident):

1. ATTRIBUTE NAMES ARE LOAD-BEARING. GAP damps `policy.encoder` parameters whose name contains
   the substring 'pro'. In their FeatureExtractor that matches `proencoder.*` (intended) AND
   `imgencoder.projection.*` / `depencoder.projection.*` — because "projection" contains "pro".
   That is almost certainly unintended by the authors, but it IS what their code does, so we
   reproduce it: our cloud branch is named `imgencoder` and keeps a `projection` head.
2. The PointNet's own final layer is named `head`, NOT `final_projection` as in our
   pointnet_diffusion.py. "final_projection" would match 'pro' and damp the cloud BACKBONE,
   which has no analogue in their setup (no ResNet parameter contains 'pro'). Renaming keeps
   the damped set faithful. The layer's shape/role is unchanged.
3. Everything else in the cloud branch — the temporal transformer settings (4 layers, 6 heads,
   head_output_size 64, mlp 256, dropout 0.1), the reshape flow, the `x[:,:,-1]` return slice —
   is copied from their ImgEncoder unchanged.
"""
import json
import numpy as np
import torch
import torch.nn as nn

# their modules (sys.path is set up by the runner before this import)
import utils
import dataset as gap_dataset
from policies.bc import BCPolicy
from policies.visual_encoder import ProEncoder
from policies.modules.transformer_modules import (
    SinusoidalPositionEncoding,
    TransformerDecoder,
)


class _PointNetXYZ(nn.Module):
    """Structural copy of gentle_manip.dppo.pointnet_diffusion.PointNetEncoderXYZ.

    Same blocks (64/128/256), same LayerNorms, same max-pool. The ONLY difference is that the
    output layer is called `head` instead of `final_projection` — see faithfulness note 2.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 256, use_layernorm: bool = True):
        super().__init__()
        block = [64, 128, 256]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block[0]),
            nn.LayerNorm(block[0]) if use_layernorm else nn.Identity(), nn.ReLU(),
            nn.Linear(block[0], block[1]),
            nn.LayerNorm(block[1]) if use_layernorm else nn.Identity(), nn.ReLU(),
            nn.Linear(block[1], block[2]),
            nn.LayerNorm(block[2]) if use_layernorm else nn.Identity(), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(block[-1], out_channels), nn.LayerNorm(out_channels))

    def forward(self, x):                    # (B, N, 3) -> (B, out)
        x = self.mlp(x)
        x = torch.max(x, 1)[0]               # permutation-invariant pooling
        return self.head(x)


class CloudEncoder(nn.Module):
    """Their ImgEncoder with resnet18+SpatialProjection replaced by PointNet+Linear.

    Structure, attribute names, transformer settings and return slice are theirs verbatim.
    """

    def __init__(self, hidden_dim, cloud_feat_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial_encoder = _PointNetXYZ(3, cloud_feat_dim)
        self.projection = nn.Linear(cloud_feat_dim, hidden_dim)   # role+name of ImgEncoder.projection
        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(self.hidden_dim)
        self.temporal_transformer = TransformerDecoder(
            input_size=self.hidden_dim,
            num_layers=4,
            num_heads=6,
            head_output_size=64,
            mlp_hidden_size=256,
            dropout=0.1,
        )

    def forward(self, cloud):                # (B, T, N, 3)
        B, T, N, C = cloud.shape
        x = cloud.reshape(-1, N, C)          # (B*T, N, 3)
        x = self.spatial_encoder(x)          # (B*T, cloud_feat_dim)
        x = self.projection(x)               # (B*T, hidden_dim)
        x = x.view(B, T, 1, -1)
        pos_emb = self.temporal_position_encoding_fn(x)
        sh = x.shape
        x = x + pos_emb.unsqueeze(1)         # (B, T, 1, E)
        self.temporal_transformer.compute_mask(x.shape)
        x = utils.join_dimensions(x, 1, 2)   # N, T*1, E
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)
        return x[:, :, -1]                   # N, T, E


class CloudFeatureExtractor(nn.Module):
    """Their FeatureExtractor with the cloud branch under the SAME attribute name (`imgencoder`)
    so GAP's 'pro' name filter selects exactly the analogous parameter set."""

    def __init__(self, hidden_dim, image=True, proprio=True, pro_dim=8, raw=False, depth=False):
        super().__init__()
        assert not depth, "arm F has a single visual modality (the point cloud)"
        self.hidden_dim = hidden_dim
        self.image, self.proprio, self.depth = image, proprio, False
        self.feature_dim = 0
        if image:
            self.imgencoder = CloudEncoder(hidden_dim)
            self.feature_dim += self.imgencoder.hidden_dim
        if proprio:
            self.proencoder = ProEncoder(hidden_dim, pro_dim, raw)   # THEIR proprio encoder
            self.feature_dim += self.proencoder.hidden_dim

    def forward(self, image=None, proprio=None, depth=None, mask_flag=None):
        modal_list = []
        if self.image:
            modal_list.append(self.imgencoder(image))
        if self.proprio:
            modal_list.append(self.proencoder(proprio))
        if mask_flag is not None and mask_flag < len(modal_list):
            modal_list[mask_flag] = torch.zeros_like(modal_list[mask_flag])
        if len(modal_list) > 1:
            return torch.cat(modal_list, dim=-1)
        return modal_list[0]


class CloudBCPolicy(BCPolicy):
    """Their BCPolicy. compute_loss is overridden ONLY to drop `B,T,C,H,W = img.shape`, an
    unused unpack that assumes a 5-D image tensor; a cloud batch is 4-D (B,T,N,3). The
    encoder/head calls are unchanged."""

    def compute_loss(self, img, proprio, gt_action, depth=None, mask_flag=None):
        hidden = self.encoder(img, proprio, depth, mask_flag)   # (B,T,E)
        return self.head.loss_fn(hidden, gt_action)


class GMCloudPhaseDataset(gap_dataset.Dataset):
    """Our npz in the batch format their WorkSpace.train() consumes.

    Yields {'img': cloud, 'proprio', 'actions', 'phase'} — the cloud rides under the 'img' key
    so their trainer needs no change at all. `states`/`actions` in the npz are ALREADY
    normalized to exactly [-1,1] (verified), which is both what our C'/D'/E arms train on and
    what their Tanh-headed DeterministicMLP expects, so no rescaling is applied.

    _build_mapping is their windowing logic, copied verbatim from PhaseDataset.
    """

    def __init__(self, data_path, batch_size=128, demo_range=(0, 2), history=5, horizon=9,
                 phase_json=None):
        super(GMCloudPhaseDataset, self).__init__(batch_size)
        d = np.load(data_path)
        tl = d["traj_lengths"]
        lo, hi = int(demo_range[0]), int(demo_range[1])
        starts = np.concatenate([[0], np.cumsum(tl)[:-1]])

        # Materialize each array ONCE. Indexing an NpzFile re-reads and decompresses the WHOLE
        # member every time, so `d["point_cloud"][s:s+L]` inside the loop would decompress 3.1 GB
        # on each of 1248 iterations. The per-trajectory slices below are then free (numpy views).
        pc_all, st_all, ac_all = d["point_cloud"], d["states"], d["actions"]
        # A numpy slice is a VIEW, so keeping per-trajectory views of a 10% subset would still pin
        # the whole 3.1 GB buffer alive. Their WorkSpace builds a second (validation) dataset over
        # demos [0, 0.1*N], so copy the needed span when it is a strict subset; for the full range
        # use the buffer directly and avoid a pointless 3.1 GB duplication.
        s0 = int(starts[lo])
        s1 = int(starts[hi - 1] + tl[hi - 1])
        if (s1 - s0) < 0.9 * len(st_all):
            pc_all = pc_all[s0:s1].copy()
            st_all = st_all[s0:s1].copy()
            ac_all = ac_all[s0:s1].copy()
        else:
            s0 = 0
        self.clouds, self.pros, self.acts = [], [], []
        for t in range(lo, hi):
            s, L = int(starts[t]) - s0, int(tl[t])
            self.clouds.append(pc_all[s:s + L])
            self.pros.append(st_all[s:s + L])
            self.acts.append(ac_all[s:s + L])

        with open(phase_json, "r") as f:
            all_phase = json.load(f)["phase"]
        assert len(all_phase) == len(tl), (
            f"phase file has {len(all_phase)} trajectories but the npz has {len(tl)} — the "
            f"phase json must come from THIS split")
        self.phase = [np.asarray(all_phase[t], np.float32).ravel() for t in range(lo, hi)]
        for i, (p, c) in enumerate(zip(self.phase, self.clouds)):
            assert len(p) == len(c), f"traj {lo+i}: phase {len(p)} != steps {len(c)}"

        self.mapping = self._build_mapping(history, horizon)

    def _build_mapping(self, history, horizon):
        """Verbatim from their PhaseDataset._build_mapping."""
        mapping = []
        traj_info = [(i, self.clouds[i].shape[0], self.phase[i]) for i in range(len(self.clouds))]
        for traj_id, traj_len, traj_phase in traj_info:
            def f(x):
                return max(0, min(x, traj_len - 1))
            for i in range(traj_len):
                img2read, pro2read, act2read = [], [], []
                for idx in range(i - history + 1, i + 1):
                    img2read.append(f(idx))
                    pro2read.append(f(idx))
                    act2read.append([f(idx + chunk) for chunk in range(horizon)])
                mapping.append([traj_id, img2read, pro2read, act2read, traj_phase[pro2read[-1]]])
        return mapping

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        traj_id, img2read, pro2read, act2read, point_phase = self.mapping[index]
        img = self.clouds[traj_id][img2read]                                  # (T, N, 3)
        pro = self.pros[traj_id][pro2read]                                    # (T, pro_dim)
        act = np.concatenate([self.acts[traj_id][chunk][np.newaxis, :] for chunk in act2read])
        return {
            "img": torch.from_numpy(np.ascontiguousarray(img)).float(),
            "proprio": torch.from_numpy(np.ascontiguousarray(pro)).float(),
            "actions": torch.from_numpy(act).float(),
            "phase": torch.tensor(point_phase).float(),
        }
