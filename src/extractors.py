import os
import sys
import copy
import math
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from multiprocessing import Pool  

sys.path.append('./inflated_convnets_pytorch')
from src.i3res import I3ResNet  

class MediaPipeExtractor:
    """Extractor for pose, face, and hand landmarks using MediaPipe."""
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
        self.face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1)

    def extract(self, video_path):
        """Extract features from a single video."""
        cap = cv2.VideoCapture(video_path)
        feats = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_res = self.pose.process(rgb)
            hand_res = self.hands.process(rgb)
            face_res = self.face.process(rgb)

            num_face_landmarks = 478
            pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_res.pose_landmarks.landmark]).flatten() if pose_res.pose_landmarks else np.zeros(33 * 3)
            face = np.array([[lm.x, lm.y, lm.z] for lm in face_res.multi_face_landmarks[0].landmark]).flatten() if face_res.multi_face_landmarks else np.zeros(num_face_landmarks * 3)
            hands_list = []
            if hand_res.multi_hand_landmarks:
                for hand_landmarks in hand_res.multi_hand_landmarks:
                    hands_list.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            hands = np.array(hands_list).flatten()
            hands = np.pad(hands, (0, 42*3 - len(hands)), 'constant')

            frame_features = np.concatenate([pose, face, hands])
            feats.append(frame_features)
        cap.release()
        return np.array(feats, dtype=np.float32)

class I3DExtractor:
    """Extractor for video features using I3D-ResNet50."""
    def __init__(self, device='cuda', clip_length=16):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.clip_length = clip_length
        resnet2d = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = I3ResNet(copy.deepcopy(resnet2d), frame_nb=self.clip_length)
        self.model.fc = nn.Identity()
        self.model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, video_path):
        """Extract I3D features from a single video."""
        cap = cv2.VideoCapture(video_path)
        frames, feats = [], []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            frames.append(frame)
            if len(frames) == self.clip_length:
                clip = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.model(clip)
                    feats.append(feat.squeeze().cpu().numpy())
                frames = []
        cap.release()
        return np.array(feats, dtype=np.float32)

def process_video(args):
    """Helper for parallel feature extraction."""
    video_path, save_dir, logger = args  
    try:
        mp_ex = MediaPipeExtractor()  
        i3d_ex = I3DExtractor()       
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        mp_feat = mp_ex.extract(video_path)
        i3d_feat = i3d_ex.extract(video_path)
        if mp_feat.shape[0] == 0 or i3d_feat.shape[0] == 0:
            logger.warning(f"Skipping {video_path} due to missing features.")
            return
        repeats = math.ceil(len(mp_feat) / len(i3d_feat))
        i3d_feat_expanded = np.repeat(i3d_feat, repeats, axis=0)[:len(mp_feat)]
        merged = np.concatenate([mp_feat, i3d_feat_expanded], axis=1)
        np.save(os.path.join(save_dir, f"{video_id}.npy"), merged)
    except Exception as e:
        logger.error(f"Error processing {video_path}: {e}")

def extract_and_save_features(video_list, save_dir, num_workers=4, logger=None):
    """Extract and save merged features, with parallel processing."""
    os.makedirs(save_dir, exist_ok=True)
    args_list = [(v, save_dir, logger) for v in video_list]  
    with Pool(num_workers) as p:
        list(tqdm(p.imap(process_video, args_list), total=len(video_list), desc="Extracting Features"))
