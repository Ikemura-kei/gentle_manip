import os
import cv2
import json
import time
import h5py
import torch
import pickle
import numpy as np
import torch.nn as nn
from collections import deque 
from torchvision import transforms


def preprocess_sequences(data, labels, weights, pad_token=0, label_pad_token=-1):
    """
    Process variable-length sequence data and corresponding timestep labels.
    
    Args:
        data: List[np.array] or List[torch.Tensor], shape [seq_len, features]
        labels: List[np.array] or List[torch.Tensor], shape [seq_len]
        weights: 
        pad_token: Data padding value (default 0)
        label_pad_token: Label padding value (default -1)
    Returns:
        padded_data: Padded data tensor (batch, max_len, features)
        padded_labels: Padded label tensor (batch, max_len)
        padded_weights
        lengths: List of actual lengths (sorted)
    """
    # Get actual length of each sample
    lengths = [len(seq) for seq in data]
    max_len = max(lengths)
    
    # Initialize padding containers
    padded_data = []
    padded_labels = []
    padded_weights = []
    
    # Pad each sample
    for i, (seq, lbl, wt) in enumerate(zip(data, labels, weights)):
        # Convert to Tensor (if input is np.array)
        seq_tensor = torch.tensor(seq) if isinstance(seq, np.ndarray) else seq
        lbl_tensor = torch.tensor(lbl) if isinstance(lbl, np.ndarray) else lbl
        wt_tensor = torch.tensor(wt) if isinstance(wt, np.ndarray) else wt
        
        # Calculate padding size
        pad_size = max_len - lengths[i]
        
        # Pad data (right padding)
        padded_seq = torch.nn.functional.pad(
            seq_tensor, 
            (0, 0, 0, pad_size),  # (left_pad, right_pad) for last dimension, feature dim not padded
            value=pad_token
        )
        padded_data.append(padded_seq)
        
        # Pad labels (right padding)
        padded_lbl = torch.nn.functional.pad(
            lbl_tensor,
            (0, pad_size),
            value=label_pad_token
        )
        padded_labels.append(padded_lbl)
    
        # Pad weights (right padding)
        padded_wt = torch.nn.functional.pad(
            wt_tensor,
            (0, pad_size),
            value=0
        )
        padded_weights.append(padded_wt)

    # Sort by length in descending order
    sorted_indices = sorted(
        range(len(lengths)), 
        key=lambda k: lengths[k], 
        reverse=True
    )
    sorted_lengths = [lengths[i] for i in sorted_indices]
    sorted_data = [padded_data[i] for i in sorted_indices]
    sorted_labels = [padded_labels[i] for i in sorted_indices]
    sorted_weights = [padded_weights[i] for i in sorted_indices]
    
    # Stack into tensors
    padded_data = torch.stack(sorted_data)  # (batch, max_len, features)
    padded_labels = torch.stack(sorted_labels)  # (batch, max_len)
    padded_weights = torch.stack(sorted_weights)  # (batch, max_len)
    
    return padded_data, padded_labels, padded_weights, sorted_lengths


class Dataset(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def get_dataloader(self, num_workers=6, shuffle=True):
        return torch.utils.data.DataLoader(self, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False, pin_memory=True)
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError


class GeneralDataset(Dataset):
    def __init__(self, data_path, batch_size=128, demo_range=[0,2], history=5, horizon=9, depth=False):
        super(GeneralDataset, self).__init__(batch_size)

        self.data_path = data_path
        self.data_source = self.data_path.split('/')[-2]

        self.demo_range = demo_range
        self.depth = depth
        self.proprio_list = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
        load_time = time.time()
        self._load_data()
        print(f"load time: {time.time()-load_time}s")
        build_time = time.time()
        self.mapping = self._build_mapping(history, horizon)
        print(f"build time: {time.time()-build_time}s")
    
    def _load_data(self):
        if self.data_source == 'metaworld':
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            data = [item[self.demo_range[0]:self.demo_range[1]] for item in data]
            imgs, pros, acts = data
            for i in range(len(imgs)):
                imgs[i] = imgs[i] / 255.0 - 0.5

            self.data = {
                'imgs': imgs,
                'pros': pros,
                'acts': acts,
            }
        elif self.data_source == 'mimicgen':
            self.data =  h5py.File(self.data_path, 'r')['data']

        return None

    def _build_mapping(self, history, horizon):
        mapping = []
        traj_info = []
        if self.data_source == 'metaworld':
            proxy_data = self.data['imgs']
            for traj_id, traj in enumerate(proxy_data):
                traj_info.append((traj_id, traj.shape[0]))

        elif self.data_source == 'mimicgen':
            for traj_id in range(self.demo_range[0], self.demo_range[1]):
                proxy_data = self.data[f'demo_{traj_id}']['obs']['agentview_image']
                traj_info.append((traj_id, proxy_data.shape[0]))

        for info in traj_info:
            traj_id, traj_len = info

            def f(x):
                return max(0, min(x, traj_len-1))
            
            for i in range(traj_len):
                img2read = []
                pro2read = []
                act2read = []
                for idx in range(i-history+1, i+1):
                    img2read.append(f(idx))
                    pro2read.append(f(idx))
                    act2read.append([f(idx+chunk) for chunk in range(horizon)])
                mapping_item = [traj_id, img2read, pro2read, act2read]
                mapping.append(mapping_item)
        return mapping

    def __len__(self):
        return len(self.mapping)
    
    def __getitem__(self, index):
        traj_id, img2read, pro2read, act2read = self.mapping[index]

        if self.data_source == 'metaworld':
            img = self.data['imgs'][traj_id][img2read]
            pro = self.data['pros'][traj_id][pro2read]
            act = np.concatenate([self.data['acts'][traj_id][chunk][np.newaxis,:] for chunk in act2read])

            item = {
                "img": torch.from_numpy(img).float(),
                "proprio": torch.from_numpy(pro).float(),
                "actions": torch.from_numpy(act).float()
            }

                
        elif self.data_source == 'mimicgen':
            def conc(data, idxs):
                data2concat = [data[idx][np.newaxis,:] for idx in idxs]
                return np.concatenate(data2concat)

            img = conc(self.data[f'demo_{traj_id}']['obs']['agentview_image'], img2read)
            img = img.transpose(0, 3, 1, 2) / 255.0 - 0.5
            robot_state = np.concatenate([self.data[f'demo_{traj_id}']['obs'][target_state] for target_state in self.proprio_list], axis=-1).astype(np.float32)
            pro = conc(robot_state, pro2read)
            act = np.concatenate([conc(self.data[f'demo_{traj_id}']['actions'], chunk)[np.newaxis,:] for chunk in act2read])

            item = {
                "img": torch.from_numpy(img).float(),
                "proprio": torch.from_numpy(pro).float(),
                "actions": torch.from_numpy(act).float()
            }

            if self.depth:
                depth = conc(self.data[f'demo_{traj_id}']['obs']['agentview_depth'], img2read)
                max_values = np.max(depth, axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
                min_values = np.min(depth, axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
                depth = (depth - min_values) / (max_values - min_values)
                depth = np.repeat(depth, 3, axis=-1)
                depth = depth.transpose(0, 3, 1, 2)
                item['depth'] = torch.from_numpy(depth).float()

        return item


class CustomLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, penalty_factor):
        """
        Args:
            pred: Model predicted probability (between 0-1)
            target: Sample labels (0 or 1)
        """
        # Create a mask to ignore samples where target is -1
        mask = target != -1
        adjusted_loss = - (target * torch.log(pred) + penalty_factor * (1 - target) * torch.log(1 - pred))
        # Apply mask to loss, ignoring positions with -1 labels
        adjusted_loss = adjusted_loss * mask.float()

        # Calculate loss according to reduction type
        if self.reduction == 'mean':
            return adjusted_loss.sum() / mask.sum()  # Calculate mean loss of non-ignored samples
        elif self.reduction == 'sum':
            return adjusted_loss.sum()  # Calculate total loss
        else:
            return adjusted_loss  # Return original loss tensor


# Define dataset
class SequenceDataset(Dataset):
    def __init__(self, data, labels, weights, lengths, batch_size=16):
        self.data = data
        self.lengths = lengths
        self.labels = labels
        self.weights = weights
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.lengths[idx], self.weights[idx]


# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=256, output_size=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        max_len = x.shape[1]
        # Pack variable-length sequences
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_x)
        # Unpack to restore padded form
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=max_len)
        # Classify each timestep
        output = self.fc(output)
        logits = self.sigmoid(output)
        return logits.squeeze(-1)  # [batch_size, seq_len]


class PhaseDataset(Dataset):
    def __init__(self, data_path, batch_size=128, demo_range=[0,2], history=5, horizon=9):
        super(PhaseDataset, self).__init__(batch_size)

        self.data_path = data_path
        self.phase_path = data_path.replace('h5df', 'json').replace('pkl', 'json')
        self.data_source = self.data_path.split('/')[-2]

        self.demo_range = demo_range
        self.proprio_list = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
        
        load_time = time.time()
        self._load_data()
        
        if not os.path.exists(self.phase_path):
            self.get_phase()
        with open(self.phase_path, 'r') as phase_file:
            self.phase = json.load(phase_file)['phase']
            
        build_time = time.time()
        self.mapping = self._build_mapping(history, horizon)

    def get_phase(self):
        # detect and bulid dataset for LSTM
        import ruptures as rpt
        detector = rpt.Pelt(model="direction")
        
        def generate_list(a, b):
            c = [0] * a
            for index in b:
                if 1 <= index <= a:
                    c[index - 1] = 1
                else:
                    raise ValueError(f"Index {index} is out of range for a list of length {a}.")
            return c
        
        pro_data = []
        pro_label = []
        pro_weights = []

        if self.data_source == 'metaworld':
            proxy_data = self.data['pros']
            for traj_id, traj_pro in enumerate(proxy_data):
                traj_pro = self.data['pros'][traj_id]
                traj_len = traj_pro.shape[0]
                pelt = detector.fit(traj_pro)
                change_points = pelt.predict(pen=4)
                change_points = [1] + change_points[:-1]
                traj_weights = [min([abs(index+1-point) for point in change_points]) for index in range(traj_len)]
                for index in range(traj_len): traj_weights[index] = 0.005 * traj_weights[index] * traj_weights[index]
                discrete_label = generate_list(traj_len, change_points)
                discrete_label = torch.tensor(discrete_label)
                
                pro_diff = np.diff(traj_pro, axis=0)
                td_pro = np.vstack([np.zeros((1, traj_pro.shape[1])), pro_diff])
                
                pro_data.append(td_pro)
                pro_label.append(discrete_label)
                pro_weights.append(np.array(traj_weights))

        elif self.data_source == 'mimicgen':
            for traj_id in range(self.demo_range[0], self.demo_range[1]):
                traj_pro = np.concatenate([self.data[f'demo_{traj_id}']['obs'][target_state] for target_state in self.proprio_list], axis=-1).astype(np.float32)
                traj_len = traj_pro.shape[0]
                pelt = detector.fit(traj_pro)
                change_points = pelt.predict(pen=4)
                change_points = [1] + change_points[:-1]
                traj_weights = [min([max(0, index+1-point) for point in change_points]) for index in range(traj_len)]
                for index in range(traj_len): traj_weights[index] = 0.05 * traj_weights[index] * traj_weights[index]
                discrete_label = generate_list(traj_len, change_points)
                discrete_label = torch.tensor(discrete_label)
                
                pro_diff = np.diff(traj_pro, axis=0)
                td_pro = np.vstack([np.zeros((1, traj_pro.shape[1])), pro_diff])

                pro_data.append(td_pro)
                pro_label.append(discrete_label)
                pro_weights.append(np.array(traj_weights))
        
        padded_data, padded_labels, padded_weights, sorted_lengths = preprocess_sequences(pro_data, pro_label, pro_weights)
        
        cp_dataset = SequenceDataset(padded_data, padded_labels, padded_weights, sorted_lengths, batch_size=32)
        cp_dataloader = cp_dataset.get_dataloader()

        phase_pred_model = LSTMModel(input_size=9, hidden_size=64, output_size=1, num_layers=2).to('cuda')
        criterion = CustomLoss()
        optimizer = torch.optim.Adam(phase_pred_model.parameters(), lr=0.01)

        # train lstm
        for epoch in range(100):
            for batch_x, batch_y, batch_len, penalty  in cp_dataloader:
                batch_x, batch_y, penalty = batch_x.to('cuda').float(), batch_y.to('cuda').float(), penalty.to('cuda').float()
                batch_x = batch_x * 1e4

                outputs = phase_pred_model(batch_x, batch_len)
                loss = criterion(outputs, batch_y, penalty)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # predict phase
        phase_pred_model.eval()
        task_phase = []
        if self.data_source == 'metaworld':
            proxy_data = self.data['pros']
            for traj_id, traj_pro in enumerate(proxy_data):
                traj_pro = self.data['pros'][traj_id]
                traj_len = traj_pro.shape[0]

                pro_diff = np.diff(traj_pro, axis=0)
                td_pro = np.vstack([np.zeros((1, traj_pro.shape[1])), pro_diff])
                td_pro = torch.from_numpy(td_pro).to("cuda").unsqueeze(0).float()
                
                with torch.no_grad():
                    logits = phase_pred_model(td_pro * 1e4, torch.tensor(traj_len).unsqueeze(0).int())
                logits = logits.squeeze(0).cpu().tolist()

                task_phase.append(logits)

        elif self.data_source == 'mimicgen':
            for traj_id in range(self.demo_range[0], self.demo_range[1]):
                traj_pro = np.concatenate([self.data[f'demo_{traj_id}']['obs'][target_state] for target_state in self.proprio_list], axis=-1).astype(np.float32)
                traj_len = traj_pro.shape[0]

                pro_diff = np.diff(traj_pro, axis=0)
                td_pro = np.vstack([np.zeros((1, traj_pro.shape[1])), pro_diff])
                td_pro = torch.from_numpy(td_pro).to("cuda").unsqueeze(0).float()
                
                traj_phase = []
                with torch.no_grad():
                    logits = phase_pred_model(td_pro * 1e4, torch.tensor(traj_len).unsqueeze(0).int())
                logits = logits.squeeze(0).cpu().tolist()

                task_phase.append(logits)

        with open(self.phase_path, 'w') as saved_phase:
            saved_data = {'phase': task_phase}
            json.dump(saved_data, saved_phase)

    def _load_data(self):
        if self.data_source == 'metaworld':
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            data = [item[self.demo_range[0]:self.demo_range[1]] for item in data[:-1]] # [imgs, pros, acts, grasp_idx]
            imgs, pros, acts = data
            for i in range(len(imgs)):
                imgs[i] = imgs[i] / 255.0 - 0.5

            self.data = {
                'imgs': imgs,
                'pros': pros,
                'acts': acts,
            }
        elif self.data_source == 'mimicgen':
            self.data =  h5py.File(self.data_path, 'r')['data']

        return None

    def _build_mapping(self, history, horizon):
        mapping = []
        traj_info = []
        if self.data_source == 'metaworld':
            proxy_data = self.data['imgs']
            for traj_id, traj in enumerate(proxy_data):
                traj_info.append((traj_id, traj.shape[0], self.phase[traj_id]))
        elif self.data_source == 'mimicgen':
            for traj_id in range(self.demo_range[0], self.demo_range[1]):         
                proxy_data = self.data[f'demo_{traj_id}']['obs']['agentview_image']
                traj_info.append((traj_id, proxy_data.shape[0], self.phase[traj_id]))

        for info in traj_info:
            traj_id, traj_len, traj_phase = info

            def f(x):
                return max(0, min(x, traj_len-1))
            
            for i in range(traj_len):
                img2read = []
                pro2read = []
                act2read = []
                for idx in range(i-history+1, i+1):
                    img2read.append(f(idx))
                    pro2read.append(f(idx))
                    act2read.append([f(idx+chunk) for chunk in range(horizon)])
                mapping_item = [traj_id, img2read, pro2read, act2read, traj_phase[pro2read[-1]]]
                mapping.append(mapping_item)
        return mapping

    def __len__(self):
        return len(self.mapping)
    
    def __getitem__(self, index):
        traj_id, img2read, pro2read, act2read, point_phase = self.mapping[index]

        if self.data_source == 'metaworld':
            img = self.data['imgs'][traj_id][img2read]
            pro = self.data['pros'][traj_id][pro2read]
            act = np.concatenate([self.data['acts'][traj_id][chunk][np.newaxis,:] for chunk in act2read])

            item = {
                "img": torch.from_numpy(img).float(),
                "proprio": torch.from_numpy(pro).float(),
                "actions": torch.from_numpy(act).float(),
                "phase": torch.tensor(point_phase).float(),
            }
        elif self.data_source == 'mimicgen':
            def conc(data, idxs):
                data2concat = [data[idx][np.newaxis,:] for idx in idxs]
                return np.concatenate(data2concat)

            img = conc(self.data[f'demo_{traj_id}']['obs']['agentview_image'], img2read)
            img = img.transpose(0, 3, 1, 2) / 255.0 - 0.5
            robot_state = np.concatenate([self.data[f'demo_{traj_id}']['obs'][target_state] for target_state in self.proprio_list], axis=-1).astype(np.float32)
            pro = conc(robot_state, pro2read)
            act = np.concatenate([conc(self.data[f'demo_{traj_id}']['actions'], chunk)[np.newaxis,:] for chunk in act2read])

            item = {
                "img": torch.from_numpy(img).float(),
                "proprio": torch.from_numpy(pro).float(),
                "actions": torch.from_numpy(act).float(),
                "phase": torch.tensor(point_phase).float(),
            }

        return item


if __name__ == '__main__':
    phase_dataset = PhaseDataset("path/to/data", demo_range=[0,100])
    print("debug done")