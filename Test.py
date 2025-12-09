# 파이프라인 시그니처 확인
import inspect
from diffusers import ZImagePipeline

print(inspect.signature(ZImagePipeline.__call__))