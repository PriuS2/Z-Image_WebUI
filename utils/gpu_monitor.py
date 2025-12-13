"""GPU 모니터링 유틸리티 - 관리자용 GPU 상태 모니터링"""

import torch
from typing import List, Dict, Any, Optional


class GPUMonitor:
    """GPU 모니터링 클래스"""
    
    def __init__(self):
        self._model_assignments: Dict[str, str] = {}  # model_name -> device
    
    @property
    def cuda_available(self) -> bool:
        """CUDA 사용 가능 여부"""
        return torch.cuda.is_available()
    
    @property
    def gpu_count(self) -> int:
        """사용 가능한 GPU 개수"""
        if not self.cuda_available:
            return 0
        return torch.cuda.device_count()
    
    def get_available_devices(self) -> List[str]:
        """사용 가능한 디바이스 목록 반환"""
        devices = ["auto", "cpu"]
        
        if self.cuda_available:
            for i in range(self.gpu_count):
                devices.append(f"cuda:{i}")
        
        # MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
        
        return devices
    
    def get_gpu_info(self, device_id: int) -> Dict[str, Any]:
        """특정 GPU의 정보 반환"""
        if not self.cuda_available or device_id >= self.gpu_count:
            return {}
        
        props = torch.cuda.get_device_properties(device_id)
        
        # 메모리 정보
        memory_allocated = torch.cuda.memory_allocated(device_id)
        memory_reserved = torch.cuda.memory_reserved(device_id)
        memory_total = props.total_memory
        memory_free = memory_total - memory_reserved
        
        return {
            "id": device_id,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "memory": {
                "total": memory_total,
                "total_gb": round(memory_total / (1024**3), 2),
                "allocated": memory_allocated,
                "allocated_gb": round(memory_allocated / (1024**3), 2),
                "reserved": memory_reserved,
                "reserved_gb": round(memory_reserved / (1024**3), 2),
                "free": memory_free,
                "free_gb": round(memory_free / (1024**3), 2),
                "usage_percent": round((memory_reserved / memory_total) * 100, 1),
            },
            "multi_processor_count": props.multi_processor_count,
        }
    
    def get_all_gpu_info(self) -> List[Dict[str, Any]]:
        """모든 GPU의 정보 반환"""
        if not self.cuda_available:
            return []
        
        gpus = []
        for i in range(self.gpu_count):
            gpu_info = self.get_gpu_info(i)
            
            # 해당 GPU에 로드된 모델 정보 추가
            loaded_models = [
                name for name, device in self._model_assignments.items()
                if device == f"cuda:{i}"
            ]
            gpu_info["loaded_models"] = loaded_models
            
            gpus.append(gpu_info)
        
        return gpus
    
    def get_system_info(self) -> Dict[str, Any]:
        """전체 시스템 GPU 정보 반환"""
        info = {
            "cuda_available": self.cuda_available,
            "gpu_count": self.gpu_count,
            "available_devices": self.get_available_devices(),
            "gpus": self.get_all_gpu_info(),
            "model_assignments": self._model_assignments.copy(),
        }
        
        # MPS 정보
        if hasattr(torch.backends, "mps"):
            info["mps_available"] = torch.backends.mps.is_available()
        else:
            info["mps_available"] = False
        
        # CUDA 버전
        if self.cuda_available:
            info["cuda_version"] = torch.version.cuda
        
        return info
    
    def get_vram_summary(self) -> str:
        """VRAM 사용량 요약 문자열 반환"""
        if not self.cuda_available:
            return "CUDA 사용 불가"
        
        summaries = []
        for i in range(self.gpu_count):
            info = self.get_gpu_info(i)
            mem = info["memory"]
            summaries.append(
                f"GPU {i}: {mem['allocated_gb']:.1f}GB / {mem['total_gb']:.1f}GB ({mem['usage_percent']:.0f}%)"
            )
        
        return " | ".join(summaries)
    
    def register_model(self, model_name: str, device: str):
        """모델이 로드된 디바이스 등록"""
        self._model_assignments[model_name] = device
    
    def unregister_model(self, model_name: str):
        """모델 등록 해제"""
        if model_name in self._model_assignments:
            del self._model_assignments[model_name]
    
    def get_model_device(self, model_name: str) -> Optional[str]:
        """모델이 로드된 디바이스 반환"""
        return self._model_assignments.get(model_name)
    
    def resolve_device(self, device: str, prefer_empty: bool = True) -> str:
        """
        디바이스 문자열을 실제 사용할 디바이스로 변환
        
        Args:
            device: "auto", "cuda:0", "cuda:1", "cpu", "mps" 등
            prefer_empty: True면 auto 선택 시 비어있는 GPU 우선 선택
        
        Returns:
            실제 사용할 디바이스 문자열
        """
        if device != "auto":
            return device
        
        # auto 모드: 최적의 디바이스 자동 선택
        if self.cuda_available:
            if not prefer_empty or self.gpu_count == 1:
                return "cuda:0"
            
            # 여러 GPU가 있으면 가장 여유로운 GPU 선택
            best_gpu = 0
            max_free = 0
            
            for i in range(self.gpu_count):
                info = self.get_gpu_info(i)
                free_memory = info["memory"]["free"]
                if free_memory > max_free:
                    max_free = free_memory
                    best_gpu = i
            
            return f"cuda:{best_gpu}"
        
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        
        return "cpu"
    
    def clear_cache(self, device: Optional[str] = None):
        """GPU 캐시 정리"""
        if not self.cuda_available:
            return
        
        if device is None:
            # 모든 GPU 캐시 정리
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif device.startswith("cuda"):
            # 특정 GPU 캐시 정리
            device_id = int(device.split(":")[1]) if ":" in device else 0
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


# 전역 인스턴스
gpu_monitor = GPUMonitor()
