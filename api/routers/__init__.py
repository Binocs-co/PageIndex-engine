from fastapi import APIRouter

from api.routers.pageindex import pageindex_router

api_router = APIRouter()
api_router.include_router(pageindex_router, prefix="/pageindex", tags=["PageIndex"])
