"""
train.py
--------
STAM-MTS 학습 루프, 검증, 모델 저장

PPT 평가 스펙 (Case Study 슬라이드):
  - Accuracy, F1-Score, Precision, Recall
  - Stratified k-fold Cross Validation (논문) →
    여기서는 train/test 고정 분할로 단순화
  - 비교 모델: kNN with DTW, Stats+Random Forest,
    Multi-channel 1D CNN, STAM-MTS (제안 모델)
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)

sys.path.append(os.path.dirname(__file__))
from dataset import get_dataloaders
from model import build_model


# ── 설정값 ──────────────────────────────────────────────────────────────
CONFIG = {
    # 데이터
    'csv_path'    : 'data/D2_data.csv',
    'batch_size'  : 32,
    'test_size'   : 0.2,
    'random_state': 42,

    # 모델 (PPT 스펙 기준)
    'num_sensors' : 18,
    'num_channels': 16,
    'kernel_size' : 3,
    'num_layers'  : 2,
    'pool_size'   : 10,
    'num_classes' : 2,
    'dropout'     : 0.3,

    # 학습
    'epochs'      : 50,
    'lr'          : 1e-3,
    'weight_decay': 1e-4,

    # 저장
    'save_dir'    : 'checkpoints',
    'model_name'  : 'stam_mts_best.pt',
}


def set_seed(seed: int = 42):
    """재현성을 위한 전역 시드 고정 (Python / NumPy / PyTorch)"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 완전 결정론적 실행 (속도 약간 저하되나 재현성 보장)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_class_weights(y_list: list, num_classes: int = 2) -> torch.Tensor:
    """
    클래스 불균형 보정용 weight 계산
    weight_c = n_total / (n_classes * n_c)
    """
    n_total = len(y_list)
    weights = []
    for c in range(num_classes):
        n_c = y_list.count(c)
        weights.append(n_total / (num_classes * n_c))
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for xs, ys in loader:
        ys = ys.to(device)
        optimizer.zero_grad()

        logits = model(xs)           # xs: list of (K, T) Tensor
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(ys.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for xs, ys in loader:
        ys = ys.to(device)
        logits = model(xs)
        loss = criterion(logits, ys)

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(ys.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec  = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    return avg_loss, acc, f1, prec, rec, all_preds, all_labels


def train(cfg: dict = CONFIG):
    set_seed(cfg['random_state'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] device: {device}")

    # ── 데이터 로드 ──────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(
        cfg['csv_path'],
        batch_size=cfg['batch_size'],
        test_size=cfg['test_size'],
        random_state=cfg['random_state'],
    )

    # ── 클래스 가중치 ─────────────────────────────────────────────────────
    y_train_all = []
    for _, ys in train_loader:
        y_train_all.extend(ys.tolist())
    class_weights = get_class_weights(y_train_all).to(device)
    print(f"[train] class weights: {class_weights.tolist()}")

    # ── 모델 / 손실함수 / 옵티마이저 ──────────────────────────────────────
    model = build_model(
        num_sensors=cfg['num_sensors'],
        num_channels=cfg['num_channels'],
        kernel_size=cfg['kernel_size'],
        num_layers=cfg['num_layers'],
        pool_size=cfg['pool_size'],
        num_classes=cfg['num_classes'],
        dropout=cfg['dropout'],
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    os.makedirs(cfg['save_dir'], exist_ok=True)
    save_path = os.path.join(cfg['save_dir'], cfg['model_name'])

    best_f1   = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
               'test_loss':  [], 'test_acc':  [], 'test_f1':  []}

    print(f"\n{'Epoch':>5} | {'Tr Loss':>8} | {'Tr Acc':>7} | {'Tr F1':>7} "
          f"| {'Te Loss':>8} | {'Te Acc':>7} | {'Te F1':>7} | {'LR':>8}")
    print("-" * 75)

    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        te_loss, te_acc, te_f1, te_prec, te_rec, _, _ = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step(te_f1)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['train_f1'].append(tr_f1)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)
        history['test_f1'].append(te_f1)

        # best 모델 저장 (test F1 기준)
        if te_f1 > best_f1:
            best_f1 = te_f1
            best_epoch = epoch
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_f1'    : best_f1,
                'config'     : cfg,
            }, save_path)
            marker = ' ← best'
        else:
            marker = ''

        elapsed = time.time() - t0
        print(f"{epoch:>5} | {tr_loss:>8.4f} | {tr_acc:>7.4f} | {tr_f1:>7.4f} "
              f"| {te_loss:>8.4f} | {te_acc:>7.4f} | {te_f1:>7.4f} "
              f"| {current_lr:.2e}{marker}")

    # ── 최종 평가 (best 모델 로드) ─────────────────────────────────────────
    print(f"\n[train] 학습 완료 | best epoch: {best_epoch} | best test F1: {best_f1:.4f}")
    print(f"[train] 모델 저장: {save_path}")

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])

    _, final_acc, final_f1, final_prec, final_rec, preds, labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n{'='*50}")
    print(f"[최종 성능 — PPT Case Study 기준 지표]")
    print(f"  Accuracy : {final_acc:.4f}")
    print(f"  F1-Score : {final_f1:.4f}")
    print(f"  Precision: {final_prec:.4f}")
    print(f"  Recall   : {final_rec:.4f}")
    print(f"{'='*50}")
    print("\n[Classification Report]")
    print(classification_report(labels, preds, target_names=['정상(0)', '불량(1)']))

    return model, history


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    train()
