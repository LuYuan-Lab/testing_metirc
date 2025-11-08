"""
独立的目标跟踪模块
专门用于推理/部署阶段的目标跟踪功能
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
except ImportError:
    print("⚠️ ultralytics not available, tracking functionality will be limited")


@dataclass
class TrackingConfig:
    """跟踪配置参数"""
    enable_tracking: bool = False           # 是否启用跟踪
    tracker_type: str = "bytetrack"         # 跟踪器类型 ["bytetrack", "botsort"] 
    track_buffer: int = 30                  # 跟踪缓冲帧数
    match_threshold: float = 0.5            # 匹配阈值
    min_box_area: float = 100              # 最小检测框面积
    track_high_thresh: float = 0.5         # 高置信度阈值
    track_low_thresh: float = 0.1          # 低置信度阈值
    new_track_thresh: float = 0.6          # 新轨迹阈值
    track_lost_thresh: int = 30            # 轨迹丢失阈值（帧数）


class VideoTracker:
    """视频目标跟踪器"""
    
    def __init__(self, config: TrackingConfig):
        """
        初始化跟踪器
        
        Args:
            config: 跟踪配置参数
        """
        self.config = config
        self.tracks: Dict[int, Dict] = {}      # 存储所有轨迹信息
        self.frame_count = 0                   # 当前帧计数
        self.last_detections: Optional[List] = None
        
        # 轨迹状态
        self.active_tracks: List[int] = []     # 活跃轨迹ID
        self.lost_tracks: List[int] = []       # 丢失轨迹ID
        self.removed_tracks: List[int] = []    # 移除轨迹ID
        
    def update(
        self, 
        detections: List, 
        frame_id: int = None,
        frame_size: Tuple[int, int] = None,
        track_confidence_threshold: float = None,
        new_track_threshold: float = None,
        track_lost_frames: int = None,
        enable_kalman_filter: bool = True,
        motion_model: str = "constant_velocity",
        association_metric: str = "iou",
        max_distance: float = 100,
        enable_track_smoothing: bool = True
    ) -> Dict[int, Dict]:
        """
        增强的跟踪更新接口，支持丰富的参数调试
        
        Args:
            detections: 检测结果，支持多种格式
            frame_id: 帧ID
            frame_size: 帧尺寸 (width, height)
            track_confidence_threshold: 跟踪置信度阈值
            new_track_threshold: 新轨迹阈值
            track_lost_frames: 轨迹丢失帧数阈值
            enable_kalman_filter: 是否启用卡尔曼滤波
            motion_model: 运动模型
            association_metric: 关联度量
            max_distance: 最大关联距离
            enable_track_smoothing: 是否启用轨迹平滑
            
        Returns:
            Dict[track_id, track_info]: 跟踪结果
        """
        if frame_id is not None:
            self.frame_count = frame_id
        else:
            self.frame_count += 1
        
        # 应用临时参数覆盖
        original_config = {}
        if track_confidence_threshold is not None:
            original_config['track_high_thresh'] = self.config.track_high_thresh
            self.config.track_high_thresh = track_confidence_threshold
        if new_track_threshold is not None:
            original_config['new_track_thresh'] = self.config.new_track_thresh
            self.config.new_track_thresh = new_track_threshold
        if track_lost_frames is not None:
            original_config['track_lost_thresh'] = self.config.track_lost_thresh
            self.config.track_lost_thresh = track_lost_frames
        
        try:
            if not self.config.enable_tracking:
                # 禁用跟踪，返回简单检测结果
                return self._format_detection_results(detections)
            
            if not detections:
                # 没有检测，更新丢失状态
                self._update_lost_tracks()
                return self._get_predicted_tracks_detailed()
            
            # 标准化检测格式
            normalized_detections = self._normalize_detections(detections)
            
            # 过滤低质量检测
            valid_detections = self._filter_detections(normalized_detections)
            
            if not valid_detections:
                self._update_lost_tracks()
                return self._get_predicted_tracks_detailed()
            
            # 数据关联
            matched_tracks, unmatched_dets, unmatched_tracks = self._associate_detections(valid_detections)
            
            # 更新匹配的轨迹
            self._update_matched_tracks(matched_tracks, valid_detections)
            
            # 处理未匹配的轨迹
            self._update_unmatched_tracks(unmatched_tracks)
            
            # 初始化新轨迹
            self._init_new_tracks(unmatched_dets, valid_detections)
            
            # 清理长时间丢失的轨迹
            self._remove_lost_tracks()
            
            return self._get_active_tracks_detailed()
            
        finally:
            # 恢复原始配置
            for key, value in original_config.items():
                setattr(self.config, key, value)
    
    def update_simple(self, detections: List[Tuple[float, float, float, float, float, int]]) -> Dict[int, Tuple[float, float, float, float]]:
        """
        简化的跟踪更新接口，保持向后兼容
        
        Args:
            detections: 检测结果列表 [(x1, y1, x2, y2, conf, class_id), ...]
            
        Returns:
            Dict[track_id, (x1, y1, x2, y2)]: 跟踪结果
        """
        self.frame_count += 1
        
        if not self.config.enable_tracking:
            # 如果禁用跟踪，直接返回检测结果（赋予简单ID）
            return {i: det[:4] for i, det in enumerate(detections)}
        
        if not detections:
            # 没有检测到目标，更新丢失状态
            self._update_lost_tracks()
            return self._get_predicted_tracks()
        
        # 过滤低质量检测
        valid_detections = self._filter_detections(detections)
        
        if not valid_detections:
            self._update_lost_tracks()
            return self._get_predicted_tracks()
        
        # 数据关联和更新
        matched_tracks, unmatched_dets, unmatched_tracks = self._associate_detections(valid_detections)
        
        # 更新匹配的轨迹
        self._update_matched_tracks(matched_tracks, valid_detections)
        
        # 处理未匹配的轨迹
        self._update_unmatched_tracks(unmatched_tracks)
        
        # 初始化新轨迹
        self._init_new_tracks(unmatched_dets, valid_detections)
        
        # 清理长时间丢失的轨迹
        self._remove_lost_tracks()
        
        return self._get_active_tracks()
    
    def _filter_detections(self, detections: List) -> List:
        """过滤低质量检测"""
        valid_dets = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            
            # 检查置信度
            if conf < self.config.track_low_thresh:
                continue
                
            # 检查框面积
            area = (x2 - x1) * (y2 - y1)
            if area < self.config.min_box_area:
                continue
                
            valid_dets.append(det)
        
        return valid_dets
    
    def _associate_detections(self, detections: List) -> Tuple[List, List, List]:
        """
        数据关联：将检测结果与现有轨迹匹配
        
        Returns:
            matched_tracks: [(track_id, det_idx), ...]
            unmatched_dets: [det_idx, ...]
            unmatched_tracks: [track_id, ...]
        """
        if not self.active_tracks:
            # 没有活跃轨迹，所有检测都是新的
            return [], list(range(len(detections))), []
        
        # 简化的匹配策略：基于IoU
        matched_tracks = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.active_tracks)
        
        for track_id in self.active_tracks:
            if track_id not in self.tracks:
                continue
                
            track_box = self.tracks[track_id]['box']
            best_match_idx = -1
            best_iou = 0.0
            
            for i, det in enumerate(detections):
                if i not in unmatched_dets:
                    continue
                    
                det_box = det[:4]
                iou = self._calculate_iou(track_box, det_box)
                
                if iou > best_iou and iou > self.config.match_threshold:
                    best_iou = iou
                    best_match_idx = i
            
            if best_match_idx >= 0:
                matched_tracks.append((track_id, best_match_idx))
                unmatched_dets.remove(best_match_idx)
                unmatched_tracks.remove(track_id)
        
        return matched_tracks, unmatched_dets, unmatched_tracks
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """计算两个框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _update_matched_tracks(self, matched_tracks: List, detections: List):
        """更新匹配成功的轨迹"""
        for track_id, det_idx in matched_tracks:
            det = detections[det_idx]
            
            if track_id not in self.tracks:
                continue
                
            # 更新轨迹信息
            self.tracks[track_id]['box'] = det[:4]
            self.tracks[track_id]['conf'] = det[4]
            self.tracks[track_id]['class_id'] = det[5]
            self.tracks[track_id]['last_update'] = self.frame_count
            self.tracks[track_id]['lost_frames'] = 0
            
            # 更新历史轨迹（可用于预测）
            if 'history' not in self.tracks[track_id]:
                self.tracks[track_id]['history'] = []
            
            self.tracks[track_id]['history'].append({
                'frame': self.frame_count,
                'box': det[:4],
                'conf': det[4]
            })
            
            # 保持历史长度限制
            if len(self.tracks[track_id]['history']) > self.config.track_buffer:
                self.tracks[track_id]['history'].pop(0)
    
    def _update_unmatched_tracks(self, unmatched_tracks: List):
        """更新未匹配的轨迹（标记为丢失）"""
        for track_id in unmatched_tracks:
            if track_id in self.tracks:
                self.tracks[track_id]['lost_frames'] += 1
                
                # 如果丢失时间过长，从活跃列表移除
                if self.tracks[track_id]['lost_frames'] > self.config.track_lost_thresh:
                    if track_id in self.active_tracks:
                        self.active_tracks.remove(track_id)
                        self.lost_tracks.append(track_id)
    
    def _init_new_tracks(self, unmatched_dets: List, detections: List):
        """初始化新轨迹"""
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            
            # 只为高置信度检测创建新轨迹
            if det[4] < self.config.new_track_thresh:
                continue
            
            # 创建新轨迹ID
            new_track_id = max(self.tracks.keys()) + 1 if self.tracks else 1
            
            # 初始化轨迹
            self.tracks[new_track_id] = {
                'box': det[:4],
                'conf': det[4],
                'class_id': det[5],
                'first_frame': self.frame_count,
                'last_update': self.frame_count,
                'lost_frames': 0,
                'history': [{
                    'frame': self.frame_count,
                    'box': det[:4],
                    'conf': det[4]
                }]
            }
            
            self.active_tracks.append(new_track_id)
    
    def _update_lost_tracks(self):
        """更新所有轨迹的丢失状态"""
        tracks_to_remove = []
        
        for track_id in self.active_tracks:
            if track_id in self.tracks:
                self.tracks[track_id]['lost_frames'] += 1
                
                if self.tracks[track_id]['lost_frames'] > self.config.track_lost_thresh:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self.active_tracks.remove(track_id)
            self.lost_tracks.append(track_id)
    
    def _remove_lost_tracks(self):
        """清理长时间丢失的轨迹"""
        tracks_to_remove = []
        
        for track_id in self.lost_tracks:
            if track_id in self.tracks:
                lost_time = self.frame_count - self.tracks[track_id]['last_update']
                
                if lost_time > self.config.track_lost_thresh * 2:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self.lost_tracks.remove(track_id)
            self.removed_tracks.append(track_id)
            if track_id in self.tracks:
                del self.tracks[track_id]
    
    def _get_predicted_tracks(self) -> Dict[int, Tuple[float, float, float, float]]:
        """获取预测的轨迹位置（当没有检测时）"""
        predicted_tracks = {}
        
        for track_id in self.active_tracks:
            if track_id in self.tracks and self.tracks[track_id]['lost_frames'] <= 5:
                # 使用最后已知位置作为预测（可以扩展为卡尔曼滤波）
                predicted_tracks[track_id] = self.tracks[track_id]['box']
        
        return predicted_tracks
    
    def _get_active_tracks(self) -> Dict[int, Tuple[float, float, float, float]]:
        """获取当前活跃的轨迹"""
        active_tracks = {}
        
        for track_id in self.active_tracks:
            if track_id in self.tracks:
                active_tracks[track_id] = self.tracks[track_id]['box']
        
        return active_tracks
    
    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """获取指定轨迹的详细信息"""
        return self.tracks.get(track_id, None)
    
    def reset(self):
        """重置跟踪器状态"""
        self.tracks.clear()
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.frame_count = 0
        
    # 新增支持方法
    def _normalize_detections(self, detections: List) -> List:
        """标准化检测格式为统一的元组格式"""
        normalized = []
        
        for det in detections:
            if isinstance(det, dict):
                # 字典格式: {'bbox': (x1,y1,x2,y2), 'confidence': conf, 'class_id': id}
                bbox = det.get('bbox', det.get('box'))
                conf = det.get('confidence', det.get('conf', 1.0))
                class_id = det.get('class_id', det.get('cls', 0))
                
                if bbox and len(bbox) == 4:
                    normalized.append((*bbox, conf, class_id))
            
            elif isinstance(det, (tuple, list)) and len(det) >= 4:
                # 元组格式: (x1, y1, x2, y2, conf, class_id)
                if len(det) == 4:
                    # 只有坐标，添加默认值
                    normalized.append((*det, 1.0, 0))
                elif len(det) == 5:
                    # 坐标+置信度，添加默认类别
                    normalized.append((*det, 0))
                else:
                    # 完整格式
                    normalized.append(det[:6])
        
        return normalized
    
    def _format_detection_results(self, detections: List) -> Dict[int, Dict]:
        """格式化检测结果为跟踪输出格式"""
        results = {}
        
        for i, det in enumerate(detections):
            if isinstance(det, dict):
                bbox = det.get('bbox', det.get('box'))
                conf = det.get('confidence', det.get('conf', 1.0))
                class_name = det.get('class', 'unknown')
                
                if bbox:
                    results[i] = {
                        'track_id': i,
                        'bbox': bbox,
                        'confidence': conf,
                        'class': class_name,
                        'state': 'detected',
                        'age': 0,
                        'hits': 1
                    }
        
        return results
    
    def _get_predicted_tracks_detailed(self) -> Dict[int, Dict]:
        """获取详细的预测轨迹信息"""
        predicted = {}
        
        for track_id in self.active_tracks:
            if track_id in self.tracks:
                track = self.tracks[track_id]
                predicted[track_id] = {
                    'track_id': track_id,
                    'bbox': track['box'],
                    'confidence': track.get('confidence', 0.5),
                    'class': track.get('class', 'person'),
                    'state': 'predicted',
                    'age': track.get('age', 0),
                    'hits': track.get('hits', 0),
                    'lost_frames': track.get('lost_frames', 0)
                }
        
        return predicted
    
    def _get_active_tracks_detailed(self) -> Dict[int, Dict]:
        """获取详细的活跃轨迹信息"""
        active = {}
        
        for track_id in self.active_tracks:
            if track_id in self.tracks:
                track = self.tracks[track_id]
                active[track_id] = {
                    'track_id': track_id,
                    'bbox': track['box'],
                    'confidence': track.get('confidence', 1.0),
                    'class': track.get('class', 'person'),
                    'state': 'active',
                    'age': track.get('age', 0),
                    'hits': track.get('hits', 1),
                    'lost_frames': 0,
                    'velocity': track.get('velocity', (0, 0)),
                    'history': track.get('history', [])
                }
        
        return active


# 便捷函数
def create_tracker(
    enable_tracking: bool = True,
    tracker_type: str = "bytetrack",
    track_buffer: int = 30,
    **kwargs
) -> VideoTracker:
    """
    创建跟踪器的便捷函数
    
    Args:
        enable_tracking: 是否启用跟踪
        tracker_type: 跟踪器类型
        track_buffer: 跟踪缓冲帧数
        **kwargs: 其他配置参数
        
    Returns:
        VideoTracker实例
    """
    config = TrackingConfig(
        enable_tracking=enable_tracking,
        tracker_type=tracker_type,
        track_buffer=track_buffer,
        **kwargs
    )
    return VideoTracker(config)


class UltralyticsTracker:
    """使用 Ultralytics 内置跟踪器的封装"""
    
    def __init__(self, model_path: str, config: TrackingConfig):
        """
        初始化 Ultralytics 跟踪器
        
        Args:
            model_path: YOLO 模型路径
            config: 跟踪配置
        """
        self.config = config
        self.model = YOLO(model_path)
        self.model.conf = 0.5  # 检测置信度
        
        # 配置跟踪器
        if config.tracker_type == "bytetrack":
            self.tracker_config = "bytetrack.yaml"
        elif config.tracker_type == "botsort":
            self.tracker_config = "botsort.yaml"
        else:
            self.tracker_config = "bytetrack.yaml"  # 默认使用 bytetrack
    
    def track(self, frame: np.ndarray, target_class: str = "person") -> Dict[int, Tuple[float, float, float, float]]:
        """
        对单帧进行目标检测和跟踪
        
        Args:
            frame: 输入帧 (H, W, C)
            target_class: 目标类别
            
        Returns:
            Dict[track_id, (x1, y1, x2, y2)]: 跟踪结果
        """
        if not self.config.enable_tracking:
            # 如果禁用跟踪，只进行检测
            results = self.model(frame, verbose=False)
            tracks = {}
            
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for i, (box, cls, conf) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
                    class_name = self.model.names[int(cls)]
                    if target_class == "all" or class_name == target_class:
                        if conf > 0.5:  # 置信度过滤
                            tracks[i] = tuple(box.cpu().numpy())
            
            return tracks
        
        # 使用内置跟踪功能
        results = self.model.track(
            frame,
            tracker=self.tracker_config,
            verbose=False,
            persist=True  # 保持跟踪状态
        )
        
        tracks = {}
        if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for box, cls, conf, track_id in zip(boxes.xyxy, boxes.cls, boxes.conf, boxes.id):
                class_name = self.model.names[int(cls)]
                if target_class == "all" or class_name == target_class:
                    if conf > self.config.track_low_thresh:
                        tracks[int(track_id)] = tuple(box.cpu().numpy())
        
        return tracks
    
    def reset(self):
        """重置跟踪器"""
        # Ultralytics 跟踪器会自动管理状态
        pass


# 导出的主要接口
__all__ = [
    'TrackingConfig',
    'VideoTracker', 
    'UltralyticsTracker'
]
