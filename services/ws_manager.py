"""WebSocket 연결 관리자"""

import asyncio
from typing import Dict, List, Optional

from fastapi import WebSocket


class SessionConnectionManager:
    """세션별 WebSocket 연결 관리"""
    
    def __init__(self):
        # session_id -> List[WebSocket]
        self._connections: Dict[str, List[WebSocket]] = {}
        self._websocket_sessions: Dict[WebSocket, str] = {}  # 역방향 매핑
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """연결 추가"""
        await websocket.accept()
        async with self._lock:
            if session_id not in self._connections:
                self._connections[session_id] = []
            self._connections[session_id].append(websocket)
            self._websocket_sessions[websocket] = session_id
    
    async def disconnect(self, websocket: WebSocket):
        """연결 제거"""
        # 지연 import로 순환 참조 방지
        from utils.queue_manager import generation_queue
        
        async with self._lock:
            session_id = self._websocket_sessions.get(websocket)
            if session_id:
                if session_id in self._connections:
                    if websocket in self._connections[session_id]:
                        self._connections[session_id].remove(websocket)
                    # 세션의 모든 연결이 끊어지면 큐에서 제거
                    if not self._connections[session_id]:
                        del self._connections[session_id]
                        # 큐에서 해당 세션 요청 제거
                        await generation_queue.remove_session_items(session_id)
                del self._websocket_sessions[websocket]
    
    async def send_to_session(self, session_id: str, message: dict):
        """특정 세션에 메시지 전송"""
        async with self._lock:
            connections = self._connections.get(session_id, [])
            for ws in connections:
                try:
                    await ws.send_json(message)
                except Exception:
                    pass
    
    async def broadcast(self, message: dict):
        """모든 연결에 브로드캐스트"""
        async with self._lock:
            for connections in self._connections.values():
                for ws in connections:
                    try:
                        await ws.send_json(message)
                    except Exception:
                        pass
    
    def get_connection_count(self) -> int:
        """총 연결 수"""
        return sum(len(conns) for conns in self._connections.values())
    
    def get_session_count(self) -> int:
        """연결된 세션 수"""
        return len(self._connections)

    def get_connected_keys(self) -> List[str]:
        """현재 연결된 키 목록 (현재는 user_{id} 형태)"""
        return list(self._connections.keys())

    async def disconnect_key(self, key: str) -> int:
        """특정 키(user_{id})의 모든 WebSocket 연결 종료"""
        async with self._lock:
            connections = list(self._connections.get(key, []))
        closed = 0
        for ws in connections:
            try:
                await ws.close(code=4000)
                closed += 1
            except Exception:
                pass
        return closed
    
    def get_session_id(self, websocket: WebSocket) -> Optional[str]:
        """WebSocket의 세션 ID 가져오기"""
        return self._websocket_sessions.get(websocket)


# 전역 싱글톤 인스턴스
ws_manager = SessionConnectionManager()
