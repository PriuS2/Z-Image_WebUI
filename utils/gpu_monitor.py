"""GPU 모니터링 유틸리티 - 관리자용 GPU 상태 모니터링"""

import torch
import time
import subprocess
from typing import List, Dict, Any, Optional


class GPUMonitor:
    """GPU 모니터링 클래스"""
    
    def __init__(self):
        self._model_assignments: Dict[str, str] = {}  # model_name -> device
        self._nvml = None
        self._nvml_available = False
        self._smi_cache: Optional[List[Dict[str, Any]]] = None
        self._smi_cache_ts: float = 0.0

        # NVML이 있으면(= NVIDIA 드라이버 환경) 실제 VRAM/사용률을 조회할 수 있다.
        # 없으면 torch.cuda 통계(프로세스/할당자 기준)로 fallback.
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_available = True
        except Exception:
            self._nvml = None
            self._nvml_available = False

    def _query_nvidia_smi(self) -> Optional[List[Dict[str, Any]]]:
        """
        nvidia-smi로 전체 GPU 사용량을 조회 (NVML/pynvml 미설치 환경 fallback)
        반환 단위:
        - memory_*: bytes
        - utilization_*: percent int
        """
        # 캐시(짧게) 사용: 관리자 패널 폴링 시 과도한 호출 방지
        now = time.time()
        if self._smi_cache is not None and (now - self._smi_cache_ts) < 1.0:
            return self._smi_cache

        try:
            # nounits: MB/% 숫자만 출력됨
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
            if not out:
                return None

            rows: List[Dict[str, Any]] = []
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue
                total_mb, used_mb, free_mb, util_gpu, util_mem = parts[:5]
                rows.append({
                    "total": int(float(total_mb)) * 1024 * 1024,
                    "used": int(float(used_mb)) * 1024 * 1024,
                    "free": int(float(free_mb)) * 1024 * 1024,
                    "util_gpu": int(float(util_gpu)),
                    "util_mem": int(float(util_mem)),
                    "source": "nvidia-smi",
                })

            self._smi_cache = rows
            self._smi_cache_ts = now
            return rows
        except Exception:
            return None
    
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

        memory_total = props.total_memory

        # (A) NVML 기반: 실제 VRAM 사용량/여유/사용률
        nvml_mem_used = None
        nvml_mem_free = None
        nvml_util_gpu = None
        nvml_util_mem = None
        util_source = "torch"

        if self._nvml_available and self._nvml is not None:
            try:
                handle = self._nvml.nvmlDeviceGetHandleByIndex(device_id)
                mem_info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                nvml_mem_used = int(mem_info.used)
                nvml_mem_free = int(mem_info.free)
                nvml_util_gpu = int(getattr(util, "gpu", 0))
                nvml_util_mem = int(getattr(util, "memory", 0))
                util_source = "nvml"
            except Exception:
                nvml_mem_used = None
                nvml_mem_free = None
                nvml_util_gpu = None
                nvml_util_mem = None

        # (A-2) NVML이 없으면 nvidia-smi로 시도
        if nvml_mem_used is None:
            smi = self._query_nvidia_smi()
            if smi and device_id < len(smi):
                nvml_mem_used = smi[device_id].get("used")
                nvml_mem_free = smi[device_id].get("free")
                nvml_util_gpu = smi[device_id].get("util_gpu")
                nvml_util_mem = smi[device_id].get("util_mem")
                util_source = smi[device_id].get("source", util_source)

        # (B) Torch 기반: 현재 프로세스(PyTorch 할당자) 기준
        memory_allocated = torch.cuda.memory_allocated(device_id)
        memory_reserved = torch.cuda.memory_reserved(device_id)
        
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
                # NVML이 있으면 실제 VRAM 기준으로 표시(양자화/bnb 포함)
                "used": nvml_mem_used if nvml_mem_used is not None else memory_reserved,
                "used_gb": round(((nvml_mem_used if nvml_mem_used is not None else memory_reserved) / (1024**3)), 2),
                "free": nvml_mem_free if nvml_mem_free is not None else max(0, memory_total - memory_reserved),
                "free_gb": round((((nvml_mem_free if nvml_mem_free is not None else max(0, memory_total - memory_reserved)) / (1024**3))), 2),
                "usage_percent": round((((nvml_mem_used if nvml_mem_used is not None else memory_reserved) / memory_total) * 100), 1),
            },
            "utilization": {
                "gpu_percent": nvml_util_gpu,
                "memory_percent": nvml_util_mem,
                "source": util_source,
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
                f"GPU {i}: {mem['used_gb']:.1f}GB / {mem['total_gb']:.1f}GB ({mem['usage_percent']:.0f}%)"
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
    
    def get_free_vram(self, device: Optional[str] = None) -> int:
        """
        특정 GPU 또는 가장 여유 있는 GPU의 여유 VRAM 반환 (bytes)
        
        Args:
            device: 특정 GPU 디바이스 (None이면 가장 여유 있는 GPU)
        
        Returns:
            여유 VRAM (bytes), CUDA 미사용 시 0
        """
        if not self.cuda_available:
            return 0
        
        if device is not None and device.startswith("cuda"):
            device_id = int(device.split(":")[1]) if ":" in device else 0
            info = self.get_gpu_info(device_id)
            return info.get("memory", {}).get("free", 0)
        
        # 가장 여유 있는 GPU 찾기
        max_free = 0
        for i in range(self.gpu_count):
            info = self.get_gpu_info(i)
            free = info.get("memory", {}).get("free", 0)
            if free > max_free:
                max_free = free
        
        return max_free
    
    def get_free_vram_gb(self, device: Optional[str] = None) -> float:
        """
        여유 VRAM을 GB 단위로 반환
        
        Args:
            device: 특정 GPU 디바이스 (None이면 가장 여유 있는 GPU)
        
        Returns:
            여유 VRAM (GB)
        """
        return self.get_free_vram(device) / (1024 ** 3)
    
    def has_enough_vram(self, required_gb: float, device: Optional[str] = None, buffer_gb: float = 1.0) -> bool:
        """
        필요한 VRAM이 충분한지 확인
        
        Args:
            required_gb: 필요한 VRAM (GB)
            device: 특정 GPU 디바이스 (None이면 가장 여유 있는 GPU)
            buffer_gb: 안전 버퍼 (GB), 기본 1GB
        
        Returns:
            충분하면 True
        """
        free_gb = self.get_free_vram_gb(device)
        return free_gb >= (required_gb + buffer_gb)
    
    def get_loaded_models(self) -> Dict[str, str]:
        """현재 로드된 모델 목록 반환 (model_name -> device)"""
        return self._model_assignments.copy()
    
    def is_model_loaded(self, model_name: str) -> bool:
        """특정 모델이 로드되어 있는지 확인"""
        return model_name in self._model_assignments


# 전역 인스턴스
gpu_monitor = GPUMonitor()
