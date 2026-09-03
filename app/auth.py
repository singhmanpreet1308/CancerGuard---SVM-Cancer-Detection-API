from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import API_KEY


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verify_api_key(
    api_key: str = Security(api_key_header)
):
    """
    Validate API key from X-API-Key request header.
    """

    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )

    return api_key