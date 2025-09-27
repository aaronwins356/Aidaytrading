"""API v1 package."""
from fastapi import APIRouter

from app.api.v1 import admin, auth, users

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router, tags=["auth"])
api_router.include_router(users.router, tags=["users"])
api_router.include_router(admin.router, tags=["admin"])

__all__ = ["api_router"]
