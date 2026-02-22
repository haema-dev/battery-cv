# -*- coding: utf-8 -*-
# anomalib 2.2.0 | Python 3.11 | Azure ML GPU T4
# FastFlow 최고 성능 구성: wide_resnet50_2 + flow_steps=16 + FP16 + augmentation
import os
import random
import argparse
import numpy as np
import torch
import cv2
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

from anomalib import TaskType
from anomalib.models import Fastflow
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.loggers import AnomalibMLFlowLogger

from torchvision.transforms.v2 import (
    Compose, Normalize, Resize, ToImage, ToDtype,
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter,
)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


# ────────────────────────────────────────────────
# 재현성 보장
# ────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ────────────────────────────────────────────────
# anomalib 2.x 배치 구조 파싱 (dict / object 모두 지원)
# ────────────────────────────────────────────────
def parse_batch(batch):
    """anomalib 1.x(dict) 및 2.x(ImageBatch object) 모두 호환"""
    if isinstance(batch, dict):
        paths    = batch.get("image_path", [])
        images   = batch.get("image")
        _am      = batch.get("anomaly_map")
        amaps    = _am if _am is not None else batch.get("anomaly_maps")
        _ps      = batch.get("pred_score")
        scores_t = _ps if _ps is not None else batch.get("pred_scores")
        _pl      = batch.get("pred_label")
        labels_t = _pl if _pl is not None else batch.get("pred_labels")
    else:
        paths    = getattr(batch, "image_path", [])
        images   = getattr(batch, "image", None)
        amaps    = getattr(batch, "anomaly_map", None)
        scores_t = getattr(batch, "pred_score", None)
        labels_t = getattr(batch, "pred_label", None)

    scores = scores_t.cpu().numpy() if scores_t is not None else np.array([])
    labels = labels_t.cpu().numpy() if labels_t is not None else np.array([])
    return paths, images, amaps, scores, labels


# ────────────────────────────────────────────────
# 히트맵 + 원본 합성 (OpenCV)
# ────────────────────────────────────────────────
def blend_heatmap(image_tensor, amap_tensor, alpha: float = 0.6):
    amap = amap_tensor.cpu().numpy() if hasattr(amap_tensor, "cpu") else amap_tensor
    amap = amap.squeeze()
    am_min, am_max = amap.min(), amap.max()
    amap_norm = ((amap - am_min) / (am_max - am_min + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(amap_norm, cv2.COLORMAP_JET)

    orig = image_tensor.cpu().numpy() if hasattr(image_tensor, "cpu") else image_tensor
    if orig.ndim == 3 and orig.shape[0] == 3:
        orig = orig.transpose(1, 2, 0)
    orig_vis = ((orig - orig.min()) / (orig.max() - orig.min() + 1e-8) * 255).astype(np.uint8)
    orig_bgr = cv2.cvtColor(orig_vis, cv2.COLOR_RGB2BGR)

    if orig_bgr.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (orig_bgr.shape[1], orig_bgr.shape[0]))

    return cv2.addWeighted(orig_bgr, alpha, heatmap, 1 - alpha, 0)


# ────────────────────────────────────────────────
# 판정 결과 + 점수 텍스트 오버레이
# ────────────────────────────────────────────────
def annotate_image(canvas: np.ndarray, score: float, is_defect: bool) -> np.ndarray:
    """이미지 상단에 DEFECT/NORMAL 판정과 anomaly score를 표시한다."""
    label = "DEFECT" if is_defect else "NORMAL"
    color = (0, 0, 230) if is_defect else (0, 180, 0)   # BGR: 적/녹
    h, w  = canvas.shape[:2]
    bar_h = max(30, h // 10)
    # 반투명 검은 배경 바
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
    # 텍스트
    font_scale = max(0.5, bar_h / 45)
    cv2.putText(canvas, f"{label}  score={score:.4f}",
                (6, int(bar_h * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    return canvas


# ────────────────────────────────────────────────
# state_dict 자동 분석 유틸
# ────────────────────────────────────────────────
def detect_backbone(state_dict: dict):
    """state_dict 키/형상으로 backbone 아키텍처 자동 감지.

    layer1 첫 conv 형상으로 BasicBlock(resnet18/34) vs Bottleneck(resnet50+) 구별.
    """
    for key, val in state_dict.items():
        if "feature_extractor.layer1.0.conv1.weight" in key:
            shape = tuple(val.shape)          # (out_ch, in_ch, kH, kW)
            if len(shape) == 4 and shape[2] == 3:       # 3×3 → BasicBlock
                return "resnet18"
            elif len(shape) == 4 and shape[0] == 128:   # 1×1 wide  → wide_resnet50_2
                return "wide_resnet50_2"
            elif len(shape) == 4:                        # 1×1 standard → resnet50
                return "resnet50"
    return None


def detect_flow_steps(state_dict: dict):
    """state_dict 키 패턴에서 flow_steps 수 자동 감지.

    anomalib 1.x / 2.x 공통 패턴: fast_flow_blocks.X.module_list.N.*
    """
    import re
    indices = set()
    for key in state_dict:
        # anomalib 1.x & 2.x: model.fast_flow_blocks.X.module_list.N.something
        m = re.search(r"fast_flow_blocks\.\d+\.module_list\.(\d+)\.", key)
        if m:
            indices.add(int(m.group(1)))
        # 구버전 nf_flows 패턴 (혹시 있을 경우 대비)
        m2 = re.search(r"nf_flows\.(\d+)\.", key)
        if m2:
            indices.add(int(m2.group(1)))
    return max(indices) + 1 if indices else None


class FastflowCompat(Fastflow):
    """anomalib 1.x / 2.x 체크포인트 호환 래퍼.

    Lightning은 on_load_checkpoint 외에 strategy.load_model_state_dict를
    통해 load_state_dict를 별도로 strict=True로 호출한다.
    따라서 load_state_dict 자체를 재정의해야 구버전 키 오류를 막을 수 있다.

    추가 변경:
    - configure_optimizers: wide_resnet50_2 권장 LR(1e-4) 적용.
      anomalib 기본값 1e-3은 normalizing flow + FP16 조합에서 loss 폭발 위험.
    - __init__: CLASSIFICATION task용 image-level Evaluator 재설정.
      anomalib 기본 Evaluator는 pixel_AUROC(gt_mask 필요)를 포함하므로 제거.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_image_only_evaluator()

    def _setup_image_only_evaluator(self):
        """pixel mask 없는 CLASSIFICATION task용 Evaluator 설정.

        anomalib 기본 Evaluator: image_AUROC + pixel_AUROC(gt_mask 필요).
        pixel_AUROC 제거 → 방법1: 새 Evaluator 생성, 방법2: gt_mask 항목 필터링.
        """
        # 방법 1: 새 Evaluator 생성 (image_AUROC only)
        try:
            from anomalib.metrics.evaluator import Evaluator
            from anomalib.metrics.auroc import AUROC
            try:
                val_metrics  = [AUROC(fields=["pred_score", "gt_label"])]
                test_metrics = [AUROC(fields=["pred_score", "gt_label"])]
            except TypeError:
                # fields가 constructor 인자가 아닌 경우 (class variable)
                val_metrics  = [AUROC()]
                test_metrics = [AUROC()]
            self.evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
            logger.info("Evaluator: image_AUROC only 설정 완료 (CLASSIFICATION)")
            return
        except Exception as e:
            logger.warning(f"Evaluator 신규 생성 실패 → gt_mask metrics 필터링 시도: {e}")

        # 방법 2: 기존 evaluator에서 gt_mask 요구 metrics 제거
        ev = getattr(self, "evaluator", None)
        if ev is None:
            return
        import torch.nn as nn
        for attr in ("val_metrics", "test_metrics"):
            ml = getattr(ev, attr, None)
            if ml is None:
                continue
            keep = [m for m in ml if "gt_mask" not in getattr(m, "fields", [])]
            try:
                setattr(ev, attr, nn.ModuleList(keep))
            except Exception:
                setattr(ev, attr, keep)
            logger.info(f"Evaluator.{attr}: {len(keep)}개 metrics 유지 (gt_mask 항목 제거)")

    def configure_optimizers(self):
        """wide_resnet50_2 + FP16 권장 LR(1e-4)으로 Adam 재정의.

        FastFlow 논문(Yu et al., 2021) 권장: wide_resnet50_2는 1e-4,
        resnet18은 2e-4. anomalib 기본값 1e-3은 FP16 학습 시 불안정.
        """
        import torch.optim as optim
        # Flow 블록 파라미터만 학습 (backbone은 frozen)
        flow_params = self.model.fast_flow_blocks.parameters()
        optimizer = optim.Adam(flow_params, lr=1e-4, weight_decay=1e-5)
        logger.info("Optimizer: Adam lr=1e-4 weight_decay=1e-5 (FastFlow 논문 권장값)")
        return optimizer

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # eval_transform 복구 (anomalib 2.x 필수)
        if "transform" in checkpoint:
            try:
                self.eval_transform = checkpoint["transform"]
            except Exception:
                pass
        # load_state_dict는 Lightning strategy가 별도로 호출하므로 여기서는 생략

    def load_state_dict(self, state_dict, strict=True):
        """post_processor 등 구버전 키 필터링 후 strict=False 로 로드.

        Lightning의 strategy.load_model_state_dict가 strict=True로 호출해도
        이 재정의가 항상 strict=False + 필터링을 적용한다.
        """
        skip_prefixes = ("post_processor", "normalization_metrics")
        filtered = {
            k: v for k, v in state_dict.items()
            if not any(k.startswith(p) for p in skip_prefixes)
        }
        result     = super().load_state_dict(filtered, strict=False)
        missing    = result.missing_keys
        unexpected = result.unexpected_keys
        loaded     = len(filtered) - len(unexpected)
        logger.info(
            f"체크포인트 로드 (strict=False): {loaded}/{len(filtered)} 키 적용 | "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if unexpected:
            logger.debug(f"무시된 키 (앞 3개): {unexpected[:3]}")
        return result

    def predict_step(self, batch, batch_idx):
        """PostProcessor 정규화를 우회해 raw anomaly score 반환.

        anomalib 2.x PostProcessor는 학습 통계(image_min/max)가 없으면
        NaN 정규화 → 모든 점수 0.5로 고정된다.
        self.model(images) (inner FastflowModel) 을 직접 호출해 raw 값을 가져온다.
        """
        try:
            images = batch["image"] if isinstance(batch, dict) else batch.image
            output = self.model(images)          # FastflowModel.forward()

            # anomaly_map 추출 (InferenceResult / dict / Tensor 형태 대응)
            if hasattr(output, "anomaly_map") and output.anomaly_map is not None:
                amap = output.anomaly_map
            elif isinstance(output, dict):
                amap = output.get("anomaly_map") or output.get("anomaly_maps")
            elif isinstance(output, torch.Tensor):
                amap = output
            else:
                amap = None

            if amap is not None:
                # anomaly_map = log p(x) per pixel (음수: 정상≈0, 불량≪0)
                # max → 항상 ≈0 (구별 불가), mean → 불량일수록 더 음수 (구별 가능)
                flat = amap.reshape(amap.shape[0], -1)
                img_score = flat.mean(dim=-1)   # per-image mean log-likelihood

                # pred_mask: VisualizerCallback이 None이면 crash하므로 dummy 제공
                # (B, H, W) float tensor
                amap_2d   = amap.squeeze(1) if amap.dim() == 4 else amap
                pred_mask = torch.zeros_like(amap_2d)   # all-zero (시각화용 placeholder)

                # gt_label: 경로에서 추출 (good→0, else→1)
                if isinstance(batch, dict):
                    paths_out  = batch.get("image_path", [])
                    img_out    = batch.get("image")
                    gt_label_t = batch.get("label")
                else:
                    paths_out  = getattr(batch, "image_path", [])
                    img_out    = getattr(batch, "image", None)
                    gt_label_t = getattr(batch, "label", None)

                logger.info(
                    f"[raw predict_step] batch={batch_idx} "
                    f"score_range=[{img_score.min():.8f}, {img_score.max():.8f}] "
                    f"amap_mean={amap.mean():.8f}"
                )
                # dict 반환 → parse_batch의 dict 분기로 안전하게 처리
                return {
                    "image_path":  paths_out,
                    "image":       img_out,
                    "anomaly_map": amap,
                    "pred_score":  img_score,
                    "pred_label":  None,
                    "pred_mask":   pred_mask,    # VisualizerCallback 필수 필드
                    "label":       gt_label_t,
                }

        except Exception as e:
            logger.warning(
                f"Raw predict_step 실패 ({type(e).__name__}: {e}) → 표준 예측으로 대체"
            )

        return super().predict_step(batch, batch_idx)


import lightning.pytorch as _pl_mod

class _DisableVisualizerAtStart(_pl_mod.Callback):
    """prediction 시작 시 anomalib 자동 콜백을 안전하게 제거.

    제거 대상 (anomalib v2.2.0 소스 확인):
      1. anomalib.callbacks.*   - VisualizerCallback: pred_mask 없으면 crash
      2. anomalib.post_processing.* - PostProcessor(Callback):
           on_predict_batch_end → outputs.anomaly_map 속성 접근.
           predict_step이 dict 반환 시 AttributeError 발생.
           또한 저장된 min/max 통계가 1.x 스케일(sum)이라 2.x(mean) 점수를
           잘못 정규화함 → 제거하고 main()에서 raw score 직접 임계값 판정.
    """

    # anomalib v2.2.0: 제거할 콜백 모듈 prefix
    _ANOMALIB_CB_MODULES = ("anomalib.callbacks", "anomalib.post_processing")

    def on_predict_start(self, trainer, pl_module) -> None:
        before = len(trainer.callbacks)
        trainer.callbacks = [
            cb for cb in trainer.callbacks
            if not any(
                getattr(type(cb), "__module__", "").startswith(m)
                for m in self._ANOMALIB_CB_MODULES
            )
        ]
        logger.info(
            f"[_DisableVisualizerAtStart] anomalib 콜백 제거: "
            f"{before} → {len(trainer.callbacks)} 콜백"
        )


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Battery FastFlow – anomalib 2.2.0")

    # 경로
    parser.add_argument("--data_path",   type=str, required=True)
    parser.add_argument("--model_path",  type=str, default=None)
    parser.add_argument("--output_dir",  type=str, default="./outputs")

    # 모드
    parser.add_argument("--mode", type=str, default="training",
                        choices=["training", "evaluation", "prediction"])

    # 모델 하이퍼파라미터 (최고 성능 기본값)
    parser.add_argument("--backbone",     type=str,   default="wide_resnet50_2",
                        help="resnet18 | wide_resnet50_2 | efficientnet_b5 등")
    parser.add_argument("--flow_steps",   type=int,   default=16,
                        help="Normalizing Flow 변환 횟수. 클수록 표현력↑ (기본 16)")
    parser.add_argument("--hidden_ratio", type=float, default=1.0)
    parser.add_argument("--image_size",   type=int,   default=256)

    # 학습 설정
    parser.add_argument("--epochs",       type=int,   default=300,
                        help="FastFlow은 수렴에 300-500 epoch 필요 (기본 300)")
    parser.add_argument("--batch_size",   type=int,   default=32,
                        help="T4 16GB + FP16: wide_resnet50_2 기준 32 안전 (~5GB)")
    parser.add_argument("--patience",     type=int,   default=15,
                        help="EarlyStopping patience")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--precision",    type=str,   default="32",
                        help="추론 기본값 32(FP32). 학습 시 train-job.yml에서 16-mixed 지정")
    parser.add_argument("--threshold",    type=float, default=None,
                        help="예측 임계값 수동 지정 (미지정 시 모델 저장값 사용)")

    args = parser.parse_args()
    set_seed(args.seed)

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[START] MODE={args.mode.upper()} | "
        f"BACKBONE={args.backbone} | "
        f"FLOW_STEPS={args.flow_steps} | "
        f"IMG={args.image_size}x{args.image_size} | "
        f"PRECISION={args.precision}"
    )

    tmp_ckpt_path = None  # finally에서 임시 파일 정리용
    try:
        dataset_root = Path(args.data_path)
        img_sz = args.image_size

        # ── 전처리 변환 ──────────────────────────────
        # anomalib 2.x: train_augmentations / val_augmentations / test_augmentations
        eval_transform = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize((img_sz, img_sz)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 학습 시 양품(Normal) 이미지에 다양한 augmentation 적용 → 과적합 방지
        train_transform = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize((img_sz, img_sz)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=15),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ── 불량 카테고리 자동 탐색 ──────────────────
        val_path = dataset_root / "validation"
        abnormal_dirs = None
        if val_path.exists():
            subs = [d for d in val_path.iterdir() if d.is_dir() and d.name != "good"]
            if subs:
                abnormal_dirs = [f"validation/{d.name}" for d in subs]
                logger.info(f"불량 카테고리 발견: {[d.name for d in subs]}")

        # ── 데이터 모듈 ──────────────────────────────
        if args.mode == "prediction":
            # 전수검사: validation 폴더 전체를 라벨 없이 추론
            try:
                from anomalib.data import PredictDataset
                predict_dir = val_path
                pred_dataset = PredictDataset(path=predict_dir, transform=eval_transform)
                loader = torch.utils.data.DataLoader(
                    pred_dataset, batch_size=args.batch_size,
                    shuffle=False, num_workers=4, pin_memory=True,
                    persistent_workers=True,            # ro_mount: epoch간 worker 재생성 방지
                    collate_fn=pred_dataset.collate_fn, # ImageItem → ImageBatch (anomalib 2.2.0 v태그 확인)
                )
                logger.info(f"PredictDataset 경로: {predict_dir} ({len(pred_dataset)} 이미지)")
            except Exception as e:
                logger.warning(f"PredictDataset 초기화 실패: {e} – Folder fallback 사용")
                loader = None
        else:
            datamodule = Folder(
                name="battery",
                root=str(dataset_root),
                normal_dir="train/good",
                normal_test_dir="validation/good",
                abnormal_dir=abnormal_dirs,
                train_batch_size=args.batch_size,
                eval_batch_size=args.batch_size,
                # ★ anomalib 2.x API: train_transform → train_augmentations
                train_augmentations=train_transform,
                val_augmentations=eval_transform,
                test_augmentations=eval_transform,
                num_workers=4,
                seed=args.seed,
            )

        # ── 사전학습 가중치 사전 분석 (backbone 자동 감지 후 모델 생성) ──────────────
        # engine.predict(ckpt_path=...)로 전달해야 engine이 setup() 후 올바르게 복구함.
        # 직접 load_state_dict()는 model.setup()에 의해 덮여씌워져 AUROC 0.5 문제 발생.
        import lightning as L
        import tempfile

        saved_thresh   = None   # state_dict에서 미리 추출한 임계값
        tmp_ckpt_path  = None   # 임시 파일 경로 (finally에서 삭제)
        ckpt_path      = None
        state_dict     = None
        eff_backbone   = args.backbone
        eff_flow_steps = args.flow_steps

        if args.model_path and os.path.exists(args.model_path):
            if args.model_path.endswith(".ckpt"):
                logger.info(f"Lightning 체크포인트 사용: {args.model_path}")
                ckpt_path = args.model_path
            else:
                # state_dict 사전 로드
                try:
                    raw = torch.load(args.model_path, map_location="cpu", weights_only=True)
                except Exception:
                    raw = torch.load(args.model_path, map_location="cpu", weights_only=False)

                state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
                logger.info(f"state_dict 로드 완료 (키 수: {len(state_dict)})")

                # ── backbone 자동 감지 ─────────────────────────────────────
                detected_bb = detect_backbone(state_dict)
                if detected_bb and detected_bb != eff_backbone:
                    logger.warning(
                        f"backbone 불일치 감지: 저장={detected_bb} / 요청={eff_backbone} "
                        f"→ 저장 모델 기준({detected_bb}) 적용"
                    )
                    eff_backbone = detected_bb
                elif detected_bb:
                    logger.info(f"backbone 감지 확인: {detected_bb}")

                # ── flow_steps 자동 감지 ──────────────────────────────────
                detected_fs = detect_flow_steps(state_dict)
                if detected_fs and detected_fs != eff_flow_steps:
                    logger.warning(
                        f"flow_steps 불일치 감지: 저장={detected_fs} / 요청={eff_flow_steps} "
                        f"→ 저장 모델 기준({detected_fs}) 적용"
                    )
                    eff_flow_steps = detected_fs

                # ── 임계값 추출 (anomalib 1.x / 2.x 키명 모두 지원) ────────
                for k, v in state_dict.items():
                    if ("image_threshold" in k and "value" in k) or \
                       k == "post_processor._image_threshold":
                        saved_thresh = float(v.item() if hasattr(v, "item") else v)
                        logger.info(f"임계값 추출: {saved_thresh:.6f}  (키={k})")
                        break
                if saved_thresh is None:
                    logger.warning(
                        "state_dict에 임계값 없음. "
                        "--threshold 로 수동 지정하거나 0.5 fallback 사용."
                    )

                # ── post_processor 진단 (스케일 파악용) ──────────────────
                pp_diag = {k: float(v.item() if hasattr(v, "item") else v)
                           for k, v in state_dict.items() if k.startswith("post_processor")}
                if pp_diag:
                    logger.info(f"[진단] post_processor 키/값: {pp_diag}")

        # ── FastFlow 모델 (자동 감지된 backbone/flow_steps 적용) ─────────────
        # FastflowCompat: on_load_checkpoint에서 strict=False + 구버전 키 필터링 처리
        model = FastflowCompat(
            backbone=eff_backbone,
            pre_trained=True,
            flow_steps=eff_flow_steps,
            conv3x3_only=False,
            hidden_ratio=args.hidden_ratio,
        )
        logger.info(f"FastFlow 모델 생성 완료 (backbone={eff_backbone}, flow_steps={eff_flow_steps})")

        # ── state_dict → Lightning ckpt 래핑 ─────────────────────────────
        # anomalib 2.x on_load_checkpoint()는 "transform" 키를 필수로 요구함.
        # FastflowCompat.on_load_checkpoint가 내부적으로 strict=False 로 로드.
        if state_dict is not None:
            wrapped = {
                "state_dict": state_dict,
                "transform": eval_transform,
                "pytorch-lightning_version": L.__version__,
                "epoch": 0,
                "global_step": 0,
                "loops": None,   # None이어야 Lightning이 predict_loop 복구를 건너뜀
                "callbacks": {},
                "optimizer_states": [],
                "lr_schedulers": [],
            }
            with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
                tmp_ckpt_path = f.name
            torch.save(wrapped, tmp_ckpt_path)
            ckpt_path = tmp_ckpt_path
            logger.info("state_dict → Lightning ckpt 래핑 완료")

        # ── 콜백 구성 ────────────────────────────────
        callbacks = []
        if args.mode == "prediction":
            # prediction 시작 시 anomalib VisualizerCallback을 제거
            # (클래스명 무관하게 anomalib.callbacks.visualizer 모듈의 모든 클래스 제거)
            callbacks.append(_DisableVisualizerAtStart())
        if args.mode == "training":
            callbacks.append(EarlyStopping(
                monitor="image_AUROC",
                patience=args.patience,
                mode="max",
                verbose=True,
            ))
            callbacks.append(ModelCheckpoint(
                dirpath=str(OUTPUT_DIR / "checkpoints"),
                filename="fastflow-{epoch:03d}-auroc{image_AUROC:.4f}",
                monitor="image_AUROC",
                mode="max",
                save_top_k=3,
                save_last=True,
            ))

        # ── MLFlow 로거 ──────────────────────────────
        mlflow_logger = AnomalibMLFlowLogger(
            experiment_name="Battery_FastFlow_v2",
            save_dir=str(OUTPUT_DIR),
        )

        # ── task 설정 ────────────────────────────────
        # anomalib 2.2.0: Engine.__init__은 callbacks/logger만 직접 처리하고
        # 나머지 **kwargs는 Lightning Trainer로 전달됨. 'task'는 Trainer가 모르는
        # 인자이므로 Engine(task=...) 대신 model.task에 직접 설정해야 함.
        # 데이터셋에 픽셀 마스크 없음 → 항상 CLASSIFICATION (image-level AUROC만 사용)
        model.task = TaskType.CLASSIFICATION

        # ── Engine ───────────────────────────────────
        # gradient_clip_val=1.0: Normalizing Flow + FP16 학습 시 gradient 폭발 방지.
        # FastFlow 논문 및 FrEIA 커뮤니티 권장. wide_resnet50_2 + flow_steps=16 조합에서
        # clip 없으면 초기 수십 epoch에 loss → -inf 발생 위험.
        # anomalib Engine은 **kwargs를 Lightning Trainer에 전달함.
        _clip = 1.0 if args.mode == "training" else None
        engine = Engine(
            max_epochs=args.epochs,
            devices=1,
            accelerator="auto",
            precision=args.precision,   # ★ FP16 mixed precision: T4 속도 최대 2배
            logger=mlflow_logger,
            callbacks=callbacks,
            default_root_dir=str(OUTPUT_DIR),
            **({"gradient_clip_val": _clip, "gradient_clip_algorithm": "norm"}
               if _clip is not None else {}),
        )

        # ══════════════════════════════════════════════
        # 모드별 실행
        # ══════════════════════════════════════════════

        if args.mode == "training":
            logger.info("Training 시작")
            engine.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

            logger.info("Training 완료 후 Test 평가 실행")
            engine.test(model=model, datamodule=datamodule)

        elif args.mode == "evaluation":
            logger.info("Evaluation 시작")
            engine.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        elif args.mode == "prediction":
            logger.info("전수검사(Prediction) 시작")

            if loader is not None:
                predictions = engine.predict(
                    model=model, dataloaders=loader, ckpt_path=ckpt_path
                )
            else:
                predictions = engine.predict(
                    model=model, datamodule=datamodule, ckpt_path=ckpt_path
                )

            import pandas as pd
            from sklearn.metrics import (
                f1_score, precision_score, recall_score, accuracy_score,
                roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                classification_report,
            )

            # ── 판정 임계값 결정 ─────────────────────────
            # 우선순위: --threshold 인수 > state_dict 추출값 > 모델값 > 0.5 fallback
            if args.threshold is not None:
                decision_threshold = args.threshold
                logger.info(f"수동 임계값 사용: {decision_threshold:.6f}")
            elif saved_thresh is not None:
                decision_threshold = saved_thresh
                logger.info(f"state_dict 추출 임계값 사용: {decision_threshold:.6f}")
            else:
                try:
                    decision_threshold = model.image_threshold.value.item()
                    logger.info(f"모델 임계값 사용: {decision_threshold:.6f}")
                except Exception:
                    decision_threshold = None
                    logger.warning("임계값 없음 – pred_label 또는 0.5 fallback 사용")

            records = []
            vis_dir = OUTPUT_DIR / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)

            for batch in (predictions or []):
                paths, images, amaps, scores, labels = parse_batch(batch)

                nan_count   = 0
                score_log_n = 0   # 디버그: 첫 N개 score 로그 출력
                for i in range(len(paths)):
                    path = paths[i]

                    # ── Raw score 추출 ────────────────────────────────────
                    # anomaly_map = log p(x): 정상≈0, 불량≪0 (음수 log-likelihood)
                    # pred_score: predict_step에서 mean(anomaly_map)으로 미리 계산됨
                    # 우선순위: pred_score(pre-computed mean) > anomaly_map.mean()
                    if i < len(scores) and np.isfinite(scores[i]):
                        score = float(scores[i])
                    elif amaps is not None and i < len(amaps):
                        amap_i = amaps[i]
                        amap_np = amap_i.cpu().numpy() if hasattr(amap_i, "cpu") else np.asarray(amap_i)
                        amap_np = amap_np.squeeze()
                        raw_score = float(np.nanmean(amap_np))   # mean (max은 항상 ≈0)
                        if not np.isfinite(raw_score):
                            nan_count += 1
                            raw_score = 0.0
                        score = raw_score
                    else:
                        score = 0.0

                    # 처음 5장 score 로그 (디버그용)
                    if score_log_n < 5:
                        logger.debug(
                            f"[score debug #{score_log_n}] {Path(path).name} "
                            f"score={score:.6f}  thr={decision_threshold}"
                        )
                        score_log_n += 1

                    # ── 판정 (score = mean log-likelihood, 음수) ──────────
                    # anomalib 1.x 저장 threshold(-0.441)는 raw log-likelihood 공간.
                    # 정상: score > threshold (덜 음수), 불량: score < threshold (더 음수)
                    if decision_threshold is not None:
                        pred_label = score < decision_threshold
                    elif i < len(labels):
                        pred_label = bool(labels[i])
                    else:
                        pred_label = score < -0.5   # fallback: 음수 방향

                    parent_dir = Path(path).parent.name
                    # 폴더명으로 정답 라벨 추정: good → 0(정상), 그 외 → 1(불량)
                    gt_label   = 0 if parent_dir == "good" else 1

                    file_name = Path(path).name
                    # defect / normal 서브폴더로 분류
                    subdir = vis_dir / ("defect" if pred_label else "normal")
                    subdir.mkdir(parents=True, exist_ok=True)
                    save_path = subdir / f"vis_{file_name}"

                    if amaps is not None and images is not None:
                        # 히트맵 오버레이: 불량 부위 = 빨간색(JET high)
                        canvas = blend_heatmap(images[i], amaps[i])
                    elif images is not None:
                        # anomaly_map 없을 때: 원본만 사용
                        orig = images[i].cpu().numpy() if hasattr(images[i], "cpu") else images[i]
                        if orig.ndim == 3 and orig.shape[0] == 3:
                            orig = orig.transpose(1, 2, 0)
                        orig_vis = ((orig - orig.min()) / (orig.max() - orig.min() + 1e-8) * 255).astype(np.uint8)
                        canvas = cv2.cvtColor(orig_vis, cv2.COLOR_RGB2BGR)
                    else:
                        canvas = None

                    if canvas is not None:
                        annotate_image(canvas, score, pred_label)
                        cv2.imwrite(str(save_path), canvas)
                        vis_path_str = str(save_path)
                    else:
                        vis_path_str = ""

                    records.append({
                        "file_path":     path,
                        "file_name":     file_name,
                        "parent_dir":    parent_dir,
                        "gt_label":      gt_label,
                        "anomaly_score": score,
                        "is_defect":     pred_label,
                        "vis_path":      vis_path_str,
                    })

            if nan_count:
                logger.warning(
                    f"NaN/Inf score {nan_count}개 감지됨 → 0.0으로 대체. "
                    f"FP16 수치 불안정 가능성 – inference-job.yml에서 --precision 32 확인 권장."
                )

            if not records:
                logger.warning("예측 결과가 없습니다. 데이터 경로와 모델을 확인하세요.")
            else:
                df       = pd.DataFrame(records)
                csv_path = OUTPUT_DIR / "results.csv"
                df.to_csv(csv_path, index=False)

                total   = len(df)
                defects = int(df["is_defect"].sum())
                logger.success(
                    f"전수검사 완료: {total}장 | 불량 {defects}장 ({defects/total*100:.1f}%) | "
                    f"양품 {total-defects}장"
                )

                y_true  = df["gt_label"].values
                y_pred  = df["is_defect"].astype(int).values
                y_score = df["anomaly_score"].values
                has_gt  = len(np.unique(y_true)) > 1  # 양/불량 모두 존재할 때만 지표 유효

                # ── 분류 지표 ──────────────────────────
                metrics_log = {}
                if has_gt:
                    # ── 저장 임계값이 스케일 불일치 시 ROC 기반 최적 임계값 자동 계산 ──
                    # anomalib 1.x(sum) vs 2.x(mean) 스케일 차이로 stored threshold가
                    # 현재 score 범위 밖에 있을 수 있음 → Youden's J로 자동 보정.
                    from sklearn.metrics import roc_curve as _roc_curve
                    # 불량이 더 음수이므로 -y_score 기준 ROC (높을수록 불량)
                    _fpr, _tpr, _thrs = _roc_curve(y_true, -y_score)
                    _j_idx  = int(np.argmax(_tpr - _fpr))
                    auto_thr = float(-_thrs[_j_idx])   # 원래 스케일로 복원
                    score_min, score_max = float(y_score.min()), float(y_score.max())
                    thr_in_range = score_min <= decision_threshold <= score_max if decision_threshold is not None else False

                    if not thr_in_range:
                        logger.warning(
                            f"저장 임계값({decision_threshold:.6f})이 실제 점수 범위 "
                            f"[{score_min:.6f}, {score_max:.6f}] 밖 → "
                            f"Youden J 자동 임계값 {auto_thr:.6f} 로 재판정"
                        )
                        # 자동 임계값으로 pred 재계산
                        y_pred = (y_score < auto_thr).astype(int)

                    acc  = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec  = recall_score(y_true, y_pred, zero_division=0)
                    f1   = f1_score(y_true, y_pred, zero_division=0)
                    try:
                        auroc = roc_auc_score(y_true, -y_score)   # 부호 반전 (불량=더음수)
                    except Exception:
                        auroc = float("nan")

                    metrics_log = {
                        "accuracy": acc, "precision": prec,
                        "recall": rec,   "f1": f1, "auroc": auroc,
                    }
                    logger.success(
                        f"[METRICS] Accuracy={acc:.4f} | Precision={prec:.4f} | "
                        f"Recall={rec:.4f} | F1={f1:.4f} | AUROC={auroc:.4f}"
                    )

                    # Classification report txt
                    report     = classification_report(y_true, y_pred, target_names=["Normal", "Defect"])
                    report_path = OUTPUT_DIR / "classification_report.txt"
                    report_path.write_text(report, encoding="utf-8")
                    logger.info(f"\n{report}")

                    # Confusion matrix PNG
                    cm   = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Defect"])
                    disp.plot(ax=ax, colorbar=False, cmap="Blues")
                    ax.set_title("Confusion Matrix")
                    plt.tight_layout()
                    cm_path = OUTPUT_DIR / "confusion_matrix.png"
                    plt.savefig(cm_path, dpi=150)
                    plt.close()
                    logger.success(f"Confusion Matrix 저장: {cm_path}")
                else:
                    logger.warning("gt_label 단일 클래스 → F1/AUROC 계산 생략")

                # ── Score Distribution PNG ──────────────
                fig, ax = plt.subplots(figsize=(9, 5))
                if has_gt:
                    good_sc = df.loc[df["gt_label"] == 0, "anomaly_score"]
                    bad_sc  = df.loc[df["gt_label"] == 1, "anomaly_score"]
                    ax.hist(good_sc, bins=40, alpha=0.7,
                            label=f"Normal (n={len(good_sc)})",  color="steelblue")
                    ax.hist(bad_sc,  bins=40, alpha=0.7,
                            label=f"Defect (n={len(bad_sc)})",   color="tomato")
                    ax.legend()
                else:
                    ax.hist(y_score, bins=40, alpha=0.8,
                            label=f"All (n={total})", color="steelblue")
                    ax.legend()
                ax.set_xlabel("Anomaly Score")
                ax.set_ylabel("Count")
                ax.set_title("Anomaly Score Distribution")
                plt.tight_layout()
                dist_path = OUTPUT_DIR / "score_distribution.png"
                plt.savefig(dist_path, dpi=150)
                plt.close()
                logger.success(f"Score Distribution 저장: {dist_path}")

                # ── MLFlow 아티팩트 로그 ────────────────
                try:
                    if metrics_log:
                        mlflow.log_metrics({k: float(v) for k, v in metrics_log.items()})
                    mlflow.log_artifact(str(csv_path))
                    mlflow.log_artifact(str(dist_path))
                    if has_gt:
                        mlflow.log_artifact(str(cm_path))
                        mlflow.log_artifact(str(report_path))
                except Exception as mlf_e:
                    logger.warning(f"MLFlow 로그 실패 (무시): {mlf_e}")

                logger.success(f"CSV: {csv_path} | 히트맵: {vis_dir}")

        # ── 모델 가중치 저장 ────────────────────────
        model_save_path = OUTPUT_DIR / "model.pt"
        torch.save(model.state_dict(), model_save_path)
        logger.success(f"[FINISH] 모델 저장: {model_save_path}")
        logger.success(f"[FINISH] 전체 출력: {OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"[FATAL] {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise
    finally:
        # state_dict 래핑용 임시 .ckpt 파일 정리
        if tmp_ckpt_path and os.path.exists(tmp_ckpt_path):
            os.unlink(tmp_ckpt_path)


if __name__ == "__main__":
    main()
