"""SQLite 데이터베이스 연결 및 초기화"""

import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Generator
import threading

from config.defaults import DATA_DIR


# 데이터베이스 파일 경로
DB_PATH = DATA_DIR / "users.db"

# 스레드별 연결 관리
_local = threading.local()


def get_connection() -> sqlite3.Connection:
    """현재 스레드의 데이터베이스 연결 가져오기 (없으면 생성)"""
    if not hasattr(_local, 'connection') or _local.connection is None:
        _local.connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.connection.row_factory = sqlite3.Row
    return _local.connection


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """데이터베이스 연결 컨텍스트 매니저"""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_database():
    """데이터베이스 초기화 - 테이블 생성"""
    # data 디렉토리 생성
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 사용자 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        
        # 인덱스 생성
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)
        """)
        
        # API 키 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(100) NOT NULL,
                key_hash VARCHAR(255) NOT NULL UNIQUE,
                key_prefix VARCHAR(12) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # API 키 인덱스
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active)
        """)
        
        conn.commit()
        print("[DB] Database initialized")


def close_connection():
    """현재 스레드의 데이터베이스 연결 닫기"""
    if hasattr(_local, 'connection') and _local.connection is not None:
        _local.connection.close()
        _local.connection = None


# 모듈 로드 시 데이터베이스 초기화
init_database()

