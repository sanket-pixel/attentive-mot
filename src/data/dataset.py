import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import configparser


class Detection(Dataset):

    def __init__(self, data_path, mode=None, transform=None):
        self.data_path = data_path
        self.video_folders = sorted([folder_name
                                     for folder_name in os.listdir(self.data_path)
                                     if "FRCNN" in folder_name])
        self.frame_size = 32
        self.max_tracks = 50
        self.height = 1080
        self.width = 1920
        self.mode = mode
        self.transform = transform
        self.video_indices = np.argsort(self.video_folders)
        self.sequence_lengths = self.get_sequence_lengths()
        self.all_detections = self.get_detection_tensor()

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, video_idx):
        random_start_timestamp = self.get_random_start_timestamp(video_idx)
        detections_from_frames = self.all_detections[torch.where((self.all_detections[:, -1] == video_idx) &
                                                                 (self.all_detections[:, 0] >= random_start_timestamp[
                                                                     1]) &
                                                                 (self.all_detections[:, 0] < random_start_timestamp[
                                                                     1] + self.frame_size))].float()
        # normalize
        detections_from_frames[:, 2] = detections_from_frames[:, 2] / self.height
        detections_from_frames[:, 3] = detections_from_frames[:, 3] / self.width
        detections_from_frames[:, 4] = detections_from_frames[:, 4] / self.height
        detections_from_frames[:, 5] = detections_from_frames[:, 5] / self.width
        # get tracks and frames
        track_ids = torch.unique(detections_from_frames[:, 1])
        frame_ids = torch.unique(detections_from_frames[:, 0])
        total_tracks = track_ids.shape[0]
        # generate random frames
        if total_tracks > self.max_tracks:
            track_ids = track_ids[torch.randperm(total_tracks)[:self.max_tracks]]
        # get detections from frames
        det_list = []
        for t in track_ids:
            det = detections_from_frames[detections_from_frames[:, 1] == t]
            det_list.append(det)
        detections_from_frames = torch.vstack(det_list)
        missing_detections = []
        for t in track_ids:
            frames = detections_from_frames[detections_from_frames[:, 1] == t][:, 0]
            common_counts = torch.unique(torch.cat([frame_ids, frames]), return_counts=True)
            missing_frames = common_counts[0][torch.where(common_counts[1] == 1)]
            if len(missing_frames) > 0:
                track = torch.ones_like(missing_frames) * t
                missing_frame_track = torch.stack([missing_frames, track])
                dummy = torch.ones(5, len(track)) * -1
                missing_detections.append(torch.vstack([missing_frame_track, dummy]).T)
        missing_detections = torch.vstack(missing_detections)
        detections_from_frames = torch.vstack([detections_from_frames, missing_detections])
        detections_from_frames = detections_from_frames[detections_from_frames[:, 1].sort()[1]]
        detections_from_frames = detections_from_frames.view(-1, self.max_tracks, 7)
        if detections_from_frames.shape[0]!=32:
            print(detections_from_frames.shape)
        detections_from_frames = detections_from_frames.sort(dim=1)[0]
        detections = detections_from_frames[:, :, 2:-1]
        frame_track = detections_from_frames[:, :, :2]
        return detections, frame_track

    def get_sequence_lengths(self):
        sequence_lengths = {}
        for folder_name in self.video_folders:
            config = configparser.ConfigParser()
            config.read(os.path.join(self.data_path, folder_name, "seqinfo.ini"))
            sequence_lengths[folder_name] = int(config["Sequence"]["seqLength"])
        sequence_lengths = torch.Tensor(list(zip(self.video_indices, list(sequence_lengths.values())))).to(torch.int64)
        return sequence_lengths

    def get_random_start_timestamp(self, video_idx):
        sequence_length = self.sequence_lengths[video_idx]  # get total frames of the video
        random_frame_index = torch.randint(sequence_length[1], (1,))  # get random start frame index
        return torch.Tensor([video_idx, random_frame_index])  # return video index and frame index

    def get_detection_tensor(self):
        ground_truth_list = []
        for i, folder in enumerate(self.video_folders):
            df = pd.read_csv(os.path.join(self.data_path, self.video_folders[i], "gt", "gt.txt"), header=None)
            df.columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "x", "y", "z"]
            df["video_id"] = i
            df.drop(["x", "y", "z"], 1, inplace=True)
            ground_truth_list.append(df)
        ground_truth_df = pd.concat(ground_truth_list)
        all_detections = torch.tensor(ground_truth_df.values)
        return all_detections