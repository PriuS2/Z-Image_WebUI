"""갤러리 API 라우터"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from utils.metadata import ImageMetadata

from routers.dependencies import (
    get_session_from_request,
    require_auth,
    set_session_cookie,
)


router = APIRouter()


@router.get("/gallery")
async def get_gallery(request: Request):
    """갤러리 이미지 목록 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    outputs_dir = session.get_outputs_dir()
    
    images = []
    if outputs_dir.exists():
        for f in sorted(outputs_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
            metadata = ImageMetadata.read_metadata(f)
            images.append({
                "filename": f.name,
                "path": f"/outputs/{session.data_id}/{f.name}",
                "metadata": metadata
            })
    
    response = JSONResponse(content={"images": images})
    set_session_cookie(response, session)
    return response


# /outputs 엔드포인트는 app.py에 남아있음 (레거시 호환 및 복잡한 라우팅)
