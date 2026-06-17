"""
model.py
--------
STAM-MTS 논문 기반 MC-DCNN 모델 구현

PPT 모델 구조 스펙:
  [Preprocessing]
    - 센서별 Min-Max Scaling (dataset.py에서 처리)

  [Feature Extraction] — 센서별 독립 브랜치 (브랜치 동형화)
    - Dilated 1D Conv (드리프트/노이즈 환경 대응)
    - Leaky ReLU (Dying ReLU Problem 방지)
    - N개 Conv Layer 반복

  [Feature Aggregation]
    - Adaptive Average Pooling 1D (가변 길이 시계열 → 고정 길이)
    - 센서별 feature map concatenation
    - FC Layer → 분류

  [Interpretability]
    - 마지막 Conv Layer에 Grad-CAM hook 등록
    - Sensor x Time 단위 Grad-CAM Score 계산 → gradcam.py에서 처리
"""

import torch
import torch.nn as nn
from typing import List


class SensorBranch(nn.Module):
    """
    단일 센서 브랜치
    Dilated 1D Conv + Leaky ReLU를 N번 쌓은 구조

    PPT 스펙:
      - 모든 브랜치 동일 구조 (브랜치 동형화)
        → 센서 간 학습/스케일 조건 동일 → Grad-CAM 공정 비교 가능
      - Dilation으로 드리프트/노이즈 환경에서 강인한 구조
      - Leaky ReLU → Dying ReLU Problem 방지
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_channels: int = 16,
        kernel_size: int = 3,
        num_layers: int = 2,
        leaky_slope: float = 0.01,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # 1, 2, 4, ... (레이어마다 dilation 증가)
            padding  = dilation * (kernel_size - 1) // 2  # 길이 보존 패딩

            in_ch  = in_channels if i == 0 else num_channels
            out_ch = num_channels

            layers += [
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                ),
                nn.LeakyReLU(negative_slope=leaky_slope),
            ]

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, T)  — 단일 센서 시계열
        Returns:
            (batch, num_channels, T)
        """
        return self.conv_layers(x)


class STAM_MTS(nn.Module):
    """
    STAM-MTS: Sensor-Time Activation Mapping for Multivariate Time Series

    구조:
      입력: list of (K, T) Tensor  [배치 내 T가 다를 수 있음]
        ↓
      센서별 독립 브랜치 (K개, 동형화)
        ↓
      Adaptive Average Pooling 1D → (batch, num_channels, pool_size)
        ↓
      Concatenation → (batch, K * num_channels * pool_size)
        ↓
      FC Layer → (batch, num_classes)

    Grad-CAM hook:
      self.branches[k].conv_layers[-2]  (마지막 Conv1d 레이어)
      → gradcam.py에서 hook 등록
    """
    def __init__(
        self,
        num_sensors: int = 18,       # K
        num_channels: int = 16,      # conv 채널 수 (c1, c2)
        kernel_size: int = 3,
        num_layers: int = 2,         # conv 레이어 수 N
        pool_size: int = 10,         # Adaptive Pooling 출력 길이
        num_classes: int = 2,
        leaky_slope: float = 0.01,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_sensors = num_sensors
        self.pool_size   = pool_size

        # ── 센서별 독립 브랜치 (동형화: 모두 동일 구조) ─────────────────
        self.branches = nn.ModuleList([
            SensorBranch(
                in_channels=1,
                num_channels=num_channels,
                kernel_size=kernel_size,
                num_layers=num_layers,
                leaky_slope=leaky_slope,
            )
            for _ in range(num_sensors)
        ])

        # ── Adaptive Average Pooling 1D ──────────────────────────────
        # 가변 길이(T) → 고정 길이(pool_size)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(pool_size)

        # ── FC Layer ─────────────────────────────────────────────────
        fc_in = num_sensors * num_channels * pool_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_in, num_classes),
        )

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: list of Tensor, 각 원소 shape = (K, T)
                   T는 샘플마다 달라도 됨
        Returns:
            logits: (batch_size, num_classes)
        """
        device = next(self.parameters()).device
        out_list = []

        for x in batch:
            x = x.to(device)  # (K, T)
            K, T = x.shape

            branch_outs = []
            for k in range(self.num_sensors):
                # (1, T) → unsqueeze → (1, 1, T) → branch → (1, num_channels, T)
                sensor_input = x[k].unsqueeze(0).unsqueeze(0)  # (1, 1, T)
                feat = self.branches[k](sensor_input)           # (1, num_channels, T)
                feat = self.adaptive_pool(feat)                 # (1, num_channels, pool_size)
                branch_outs.append(feat)

            # (1, K * num_channels * pool_size)
            concat = torch.cat(branch_outs, dim=1).view(1, -1)
            out_list.append(concat)

        # (batch_size, K * num_channels * pool_size)
        out = torch.cat(out_list, dim=0)

        return self.classifier(out)  # (batch_size, num_classes)

    def get_last_conv_layers(self) -> List[nn.Conv1d]:
        """
        Grad-CAM hook 등록용: 각 브랜치의 마지막 Conv1d 레이어 반환
        """
        last_convs = []
        for branch in self.branches:
            # conv_layers: [Conv1d, LeakyReLU, Conv1d, LeakyReLU, ...]
            # 마지막 Conv1d = index -2
            last_convs.append(branch.conv_layers[-2])
        return last_convs


def build_model(
    num_sensors: int = 18,
    num_channels: int = 16,
    kernel_size: int = 3,
    num_layers: int = 2,
    pool_size: int = 10,
    num_classes: int = 2,
    dropout: float = 0.3,
) -> STAM_MTS:
    """모델 생성 헬퍼 함수"""
    model = STAM_MTS(
        num_sensors=num_sensors,
        num_channels=num_channels,
        kernel_size=kernel_size,
        num_layers=num_layers,
        pool_size=pool_size,
        num_classes=num_classes,
        dropout=dropout,
    )
    return model


# ── 동작 확인용 ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.append('src')
    from dataset import get_dataloaders

    # 데이터 로드
    train_loader, _ = get_dataloaders('data/D2_data.csv', batch_size=4)
    xs, ys = next(iter(train_loader))

    # 모델 생성
    model = build_model()
    print("[model 구조]")
    print(model)

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n총 파라미터 수: {total_params:,}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(xs)

    print(f"\n[forward pass 확인]")
    print(f"  입력 배치 크기   : {len(xs)}")
    print(f"  x[0].shape      : {xs[0].shape}  (K, T)")
    print(f"  출력 logits shape: {logits.shape}  (batch, num_classes)")
    print(f"  logits 예시      : {logits[0].tolist()}")

    # Grad-CAM hook 대상 레이어 확인
    last_convs = model.get_last_conv_layers()
    print(f"\n[Grad-CAM hook 대상]")
    print(f"  브랜치 수        : {len(last_convs)}")
    print(f"  레이어 타입      : {type(last_convs[0])}")
