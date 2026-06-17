"""
dataset.py
----------
STAM-MTS 논문 기반 데이터 로딩 및 전처리 모듈

PPT 전처리 스펙:
  - 센서별 Min-Max Scaling (센서 간 공정한 Grad-CAM 비교를 위해)
  - MaterialID 단위로 묶어 (K, T) 텐서 구성
    K = 센서(피처) 수, T = 시계열 길이 (가변)
  - Step1 + Step2를 시간축으로 이어붙여 하나의 시계열로 구성
  - train/test: is_test 컬럼 무시, MaterialID 단위 stratified 8:2 split
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List


# 데이터에서 사용할 피처 컬럼 (feature_8, 9 제거된 상태)
FEATURE_COLS = [
    'feature_1',  'feature_2',  'feature_3',  'feature_4',
    'feature_5',  'feature_6',  'feature_7',  'feature_10',
    'feature_11', 'feature_12', 'feature_13', 'feature_14',
    'feature_15', 'feature_16', 'feature_17', 'feature_18',
    'feature_19', 'feature_20'
]


def load_and_preprocess(
    csv_path: str,
    feature_cols: List[str] = FEATURE_COLS,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    """
    CSV 로드 → MaterialID 단위 stratified split
    → train 기준 센서별 Min-Max Scaling → (K, T) 시계열 구성

    Returns:
        X_train: list of (K, T) ndarray
        y_train: list of int (0=정상, 1=불량)
        X_test:  list of (K, T) ndarray
        y_test:  list of int
    """
    df = pd.read_csv(csv_path)

    # Step1 → Step2 순서 보장
    df = df.sort_values(['MaterialID', 'StepID', 'duration_ms']).reset_index(drop=True)

    # ── MaterialID 단위 stratified split ───────────────────────────────
    mat_info = df.groupby('MaterialID')['target'].first().reset_index()

    train_ids, test_ids = train_test_split(
        mat_info['MaterialID'],
        test_size=test_size,
        stratify=mat_info['target'],
        random_state=random_state,
    )
    train_ids = set(train_ids)
    test_ids  = set(test_ids)
    # ───────────────────────────────────────────────────────────────────

    # ── 센서별 Min-Max Scaling (train만으로 fit) ───────────────────────
    train_df = df[df['MaterialID'].isin(train_ids)]
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    df[feature_cols] = scaler.transform(df[feature_cols])
    # ───────────────────────────────────────────────────────────────────

    # MaterialID 단위로 (K, T) 시계열 구성
    X_train, y_train = [], []
    X_test,  y_test  = [], []

    for mat_id, group in df.groupby('MaterialID'):
        x = group[feature_cols].values.T.astype(np.float32)  # (K, T)
        y = int(group['target'].iloc[0])

        if mat_id in train_ids:
            X_train.append(x)
            y_train.append(y)
        else:
            X_test.append(x)
            y_test.append(y)

    # 결과 요약
    print(f"[dataset] train: {len(X_train)}개 "
          f"(정상 {y_train.count(0)} | 불량 {y_train.count(1)} | "
          f"불량 비율 {y_train.count(1)/len(y_train)*100:.1f}%)")
    print(f"[dataset] test : {len(X_test)}개 "
          f"(정상 {y_test.count(0)} | 불량 {y_test.count(1)} | "
          f"불량 비율 {y_test.count(1)/len(y_test)*100:.1f}%)")
    print(f"[dataset] 피처(K): {X_train[0].shape[0]} | "
          f"시계열 길이 범위(T): "
          f"{min(x.shape[1] for x in X_train)}~{max(x.shape[1] for x in X_train)}")

    return X_train, y_train, X_test, y_test


class STAMMTSDataset(Dataset):
    """
    STAM-MTS PyTorch Dataset
    가변 길이 시계열을 그대로 보유 (패딩 없음)
    → Adaptive Pooling이 FC 직전에 길이를 통일

    __getitem__ 반환:
        x: Tensor (K, T)  — 센서 x 시간
        y: Tensor scalar  — 레이블 (0=정상, 1=불량)
    """
    def __init__(self, X: List[np.ndarray], y: List[int]):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.X[idx], dtype=torch.float32)  # (K, T)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return x, label


def collate_fn(batch):
    """
    가변 길이 배치 처리용 collate_fn
    시계열 길이(T)가 샘플마다 다를 수 있으므로 list로 묶어 반환
    → model 내부 Adaptive Pooling에서 고정 길이로 통일
    """
    xs, ys = zip(*batch)
    return list(xs), torch.tensor(ys, dtype=torch.long)


def get_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    train / test DataLoader 반환
    """
    X_train, y_train, X_test, y_test = load_and_preprocess(
        csv_path, test_size=test_size, random_state=random_state
    )

    train_dataset = STAMMTSDataset(X_train, y_train)
    test_dataset  = STAMMTSDataset(X_test,  y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, test_loader


# ── 동작 확인용 ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else '../data/D2_data.csv'

    train_loader, test_loader = get_dataloaders(csv_path, batch_size=8)

    xs, ys = next(iter(train_loader))
    print(f"\n[batch 확인]")
    print(f"  배치 크기     : {len(xs)}")
    print(f"  x[0].shape   : {xs[0].shape}  (K, T)")
    print(f"  labels       : {ys}")
    print(f"  x[0] 값 범위 : {xs[0].min():.4f} ~ {xs[0].max():.4f}  (Min-Max 스케일링 확인)")
