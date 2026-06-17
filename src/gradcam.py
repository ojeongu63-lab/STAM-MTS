"""
gradcam.py
----------
STAM-MTS Grad-CAM 기반 해석 가능성 모듈

PPT 9, 10슬라이드 스펙:
  [Interpretability]
  1. 마지막 Conv Layer에 forward/backward hook 등록
  2. gradient × feature map → 센서별 × 시점별 CAM Score
  3. Feature_Importance = Σ_t |CAM_Score(sensor, t)|

  [출력]
  - 센서 × 시간 Grad-CAM 히트맵 (PPT 8슬라이드)
  - 센서별 Feature Importance 바차트 (PPT 9슬라이드, Positive/Negative 분리)

분석 대상 클래스(target_class)는 기본 1(불량)로 고정.
"불량 판정의 근거가 어느 센서·어느 시점인가"를 보는 것이 목적이므로
모델 예측과 무관하게 항상 불량 관점에서 해석한다.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from typing import List, Dict, Tuple, Optional

sys.path.append(os.path.dirname(__file__))
from model import STAM_MTS


def _set_korean_font():
    candidates = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumSquareR.ttf',
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        'C:/Windows/Fonts/malgun.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            matplotlib.rcParams['font.family'] = prop.get_name()
            matplotlib.rcParams['axes.unicode_minus'] = False
            return
    print("[gradcam] 한글 폰트를 찾을 수 없습니다. 한글이 깨질 수 있습니다.")

_set_korean_font()


FEATURE_NAMES = [
    'feature_1',  'feature_2',  'feature_3',  'feature_4',
    'feature_5',  'feature_6',  'feature_7',  'feature_10',
    'feature_11', 'feature_12', 'feature_13', 'feature_14',
    'feature_15', 'feature_16', 'feature_17', 'feature_18',
    'feature_19', 'feature_20'
]


class GradCAM:
    """STAM-MTS Grad-CAM: 각 센서 브랜치 마지막 Conv1d에 hook 등록"""

    def __init__(self, model: STAM_MTS):
        self.model = model
        self.activations: Dict[int, torch.Tensor] = {}
        self.gradients: Dict[int, torch.Tensor] = {}
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        for k, layer in enumerate(self.model.get_last_conv_layers()):
            self._handles.append(layer.register_forward_hook(self._fwd(k)))
            self._handles.append(layer.register_full_backward_hook(self._bwd(k)))

    def _fwd(self, k):
        def hook(m, inp, out): self.activations[k] = out.detach()
        return hook

    def _bwd(self, k):
        def hook(m, gin, gout): self.gradients[k] = gout[0].detach()
        return hook

    def compute(self, x: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """단일 샘플 (K,T) → Grad-CAM Score (K,T). 양수=target방향, 음수=반대"""
        self.model.eval()
        self.activations.clear()
        self.gradients.clear()
        logits = self.model([x])
        self.model.zero_grad()
        logits[0, target_class].backward()

        cam_list = []
        for k in range(x.shape[0]):
            act = self.activations[k]
            grad = self.gradients[k]
            weights = grad.mean(dim=2, keepdim=True)
            cam_k = (weights * act).sum(dim=1)
            cam_list.append(cam_k.squeeze(0).cpu().numpy())
        return np.stack(cam_list, axis=0)

    @staticmethod
    def feature_importance(cam):
        return np.abs(cam).sum(axis=1)

    @staticmethod
    def signed_importance(cam):
        pos = np.clip(cam, 0, None).sum(axis=1)
        neg = np.abs(np.clip(cam, None, 0)).sum(axis=1)
        return pos, neg

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            logits = self.model([x])
            probs = torch.softmax(logits, dim=1)
        return logits.argmax(dim=1).item(), probs[0, 1].item()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _build_title(info, short=False):
    true_str = '불량' if info['true_label'] == 1 else '정상'
    pred_str = '불량' if info['pred'] == 1 else '정상'
    mark = 'O' if info['correct'] else 'X'
    if short:
        return f"idx {info['index']} | 실제 {true_str}/예측 {pred_str}"
    return (f"idx {info['index']}  |  실제: {true_str}({info['true_label']}) "
            f"/ 예측: {pred_str}({info['pred']}) [{mark}]  |  "
            f"불량확률: {info['p_defect']:.3f}  |  {info.get('desc','')}")


def plot_cam_heatmap(cam_matrix, raw_signal, info,
                     feature_names=FEATURE_NAMES, save_path=None):
    K, T = cam_matrix.shape
    fig, axes = plt.subplots(2, 1, figsize=(14, 6),
                             gridspec_kw={'height_ratios': [1.5, 1]})
    ax0 = axes[0]
    for k in range(K):
        ax0.plot(raw_signal[k], alpha=0.4, linewidth=0.8)
    ax0.set_title(_build_title(info), fontsize=11)
    ax0.set_ylabel("Raw Signal (Min-Max Scaled)")
    ax0.set_xlim(0, T - 1)
    ax0.grid(axis='x', linestyle='--', alpha=0.3)

    ax1 = axes[1]
    vmax = np.abs(cam_matrix).max() or 1.0
    im = ax1.imshow(cam_matrix, aspect='auto', cmap='RdBu_r',
                    vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax1.set_yticks(range(K))
    ax1.set_yticklabels(feature_names, fontsize=7)
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Sensor (Feature)")
    plt.colorbar(im, ax=ax1, label='Grad-CAM Score (빨강=불량방향, 파랑=정상방향)')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[gradcam] 히트맵 저장: {save_path}")
    plt.close()


def plot_feature_importance(pos, neg, info,
                            feature_names=FEATURE_NAMES, top_k=5, save_path=None):
    importance = pos + neg
    top_idx = np.argsort(importance)[::-1][:top_k]
    top_names = [feature_names[i] for i in top_idx]
    top_pos, top_neg = pos[top_idx], neg[top_idx]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(top_k)[::-1]
    ax.barh(y_pos, top_pos, color='#A32D2D', alpha=0.9,
            edgecolor='white', linewidth=0.5, label='Positive (불량 방향)')
    ax.barh(y_pos, top_neg, left=top_pos, color='#185FA5', alpha=0.9,
            edgecolor='white', linewidth=0.5, label='Negative (정상 방향, |절댓값|)')

    max_total = (top_pos + top_neg).max()
    for yi, p, n in zip(y_pos, top_pos, top_neg):
        if p > max_total * 0.03:
            ax.text(p/2, yi, f'{p:.2f}', va='center', ha='center', fontsize=8, color='white')
        if n > max_total * 0.03:
            ax.text(p+n/2, yi, f'{n:.2f}', va='center', ha='center', fontsize=8, color='white')
        ax.text(p+n+max_total*0.01, yi, f'Σ={p+n:.3f}', va='center', fontsize=8.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=10)
    ax.set_xlabel("Raw Grad-CAM Score Integral (Sum of Absolute Values)", fontsize=9)
    ax.set_title(f"Top {top_k} Variable Importance  |  {_build_title(info, short=True)}", fontsize=11)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[gradcam] 중요도 차트 저장: {save_path}")
    plt.close()


def curate_samples(model, test_loader, device):
    """모델의 다양한 면을 보여주는 5개 샘플 자동 선정"""
    gradcam = GradCAM(model)
    model.to(device)
    records, idx = [], 0
    for xs, ys in test_loader:
        for x, y in zip(xs, ys):
            pred, p_defect = gradcam.predict(x)
            true = int(y.item())
            records.append({'index': idx, 'x': x, 'true_label': true,
                            'pred': pred, 'p_defect': p_defect,
                            'correct': pred == true})
            idx += 1
    gradcam.remove_hooks()

    cd = sorted([r for r in records if r['true_label']==1 and r['correct']], key=lambda r: -r['p_defect'])
    cn = sorted([r for r in records if r['true_label']==0 and r['correct']], key=lambda r: r['p_defect'])
    fp = sorted([r for r in records if r['true_label']==0 and not r['correct']], key=lambda r: -r['p_defect'])
    fn = sorted([r for r in records if r['true_label']==1 and not r['correct']], key=lambda r: r['p_defect'])

    sel = []
    if len(cd) >= 1: sel.append({**cd[0], 'desc': '잘 맞춘 불량 (고신뢰)'})
    if len(cd) >= 2: sel.append({**cd[1], 'desc': '잘 맞춘 불량 (일관성 확인)'})
    if len(cn) >= 1: sel.append({**cn[0], 'desc': '잘 맞춘 정상 (대조군)'})
    if len(fp) >= 1: sel.append({**fp[0], 'desc': '오분류 FP (정상→불량)'})
    if len(fn) >= 1:
        sel.append({**fn[0], 'desc': '오분류 FN (불량→정상, 가장 위험)'})
    elif len(fp) >= 2:
        sel.append({**fp[1], 'desc': '오분류 FP #2'})
    return sel[:5]


def run_analysis(model, test_loader, device, target_class=1,
                 output_dir='outputs/gradcam', samples=None, feature_names=FEATURE_NAMES):
    """큐레이션된 샘플(없으면 자동)에 대해 Grad-CAM 분석"""
    if samples is None:
        print("[gradcam] 5개 대표 샘플 자동 큐레이션 중...")
        samples = curate_samples(model, test_loader, device)

    gradcam = GradCAM(model)
    model.to(device)
    results = []

    print(f"\n{'='*70}")
    print(f"[Grad-CAM 분석] target_class={target_class} (불량 관점 해석)")
    print(f"{'='*70}")

    for i, s in enumerate(samples, 1):
        x = s['x'].to(device)
        cam = gradcam.compute(x, target_class=target_class)
        pos, neg = gradcam.signed_importance(cam)
        importance = pos + neg
        info = {k: s[k] for k in ['index','true_label','pred','p_defect','correct','desc']}
        sid = f"sample{i}_idx{s['index']}"

        plot_cam_heatmap(cam, x.detach().cpu().numpy(), info,
                         feature_names=feature_names,
                         save_path=os.path.join(output_dir, f"{sid}_heatmap.png"))
        plot_feature_importance(pos, neg, info, feature_names=feature_names,
                                save_path=os.path.join(output_dir, f"{sid}_importance.png"))

        results.append({'info': info, 'cam_matrix': cam, 'pos': pos, 'neg': neg, 'importance': importance})

        top5 = np.argsort(importance)[::-1][:5]
        print(f"\n[{i}] {s['desc']}  (idx={s['index']})")
        print(f"    실제={s['true_label']} 예측={s['pred']} "
              f"불량확률={s['p_defect']:.3f} {'정답' if s['correct'] else '오분류'}")
        print(f"    Top5 센서: " + ", ".join(
            f"{feature_names[j]}({importance[j]:.2f})" for j in top5))

    gradcam.remove_hooks()
    print(f"\n[gradcam] 분석 완료 — {len(results)}개 샘플, 저장: {output_dir}/")
    return results


def analyze_index(model, test_loader, device, sample_index,
                  target_class=1, output_dir='outputs/gradcam', feature_names=FEATURE_NAMES):
    """특정 인덱스 샘플 하나만 분석"""
    gradcam = GradCAM(model)
    model.to(device)
    found, idx = None, 0
    for xs, ys in test_loader:
        for x, y in zip(xs, ys):
            if idx == sample_index:
                pred, p_defect = gradcam.predict(x)
                found = {'index': idx, 'x': x, 'true_label': int(y.item()),
                         'pred': pred, 'p_defect': p_defect,
                         'correct': pred == int(y.item()), 'desc': '사용자 지정'}
                break
            idx += 1
        if found: break
    gradcam.remove_hooks()

    if found is None:
        print(f"[gradcam] index {sample_index} 없음")
        return None
    res = run_analysis(model, test_loader, device, target_class=target_class,
                       output_dir=output_dir, samples=[found], feature_names=feature_names)
    return res[0] if res else None


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    from dataset import get_dataloaders
    from model import build_model
    from train import set_seed

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)

    ckpt_path = 'checkpoints/stam_mts_best.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"[gradcam] 체크포인트 로드: {ckpt_path}")
    else:
        print("[gradcam] 체크포인트 없음 — 랜덤 가중치로 동작 확인")

    _, test_loader = get_dataloaders('data/D2_data.csv', batch_size=8)

    # 포트폴리오용 큐레이션 분석 (대표 샘플 5개 자동 선정, 불량 관점)
    run_analysis(model, test_loader, device, target_class=1)

    # 특정 샘플만 보고 싶을 때:
    # analyze_index(model, test_loader, device, sample_index=8, target_class=1)
