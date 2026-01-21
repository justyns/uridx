"""Single-user OAuth provider for uridx.

A minimal, self-contained OAuth 2.1 provider designed for personal use.
Stores all state in memory - suitable for single-instance deployments.
"""

import hashlib
import hmac
import secrets
import time
from urllib.parse import urlencode

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from mcp.server.auth.provider import (
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
)
from pydantic import AnyHttpUrl

from fastmcp.server.auth.auth import AccessToken, OAuthProvider


class SingleUserOAuthProvider(OAuthProvider):
    """OAuth provider for single-user, self-contained authentication.

    Features:
    - Auto-approves any client registration (DCR)
    - Simple password-based authorization
    - In-memory token storage
    - No external dependencies
    """

    def __init__(
        self,
        base_url: str,
        password: str,
        token_lifetime: int = 3600,
        refresh_token_lifetime: int = 86400 * 30,
    ):
        """Initialize the OAuth provider.

        Args:
            base_url: Public URL of the server (e.g., https://example.com)
            password: Password for authorization
            token_lifetime: Access token lifetime in seconds (default 1 hour)
            refresh_token_lifetime: Refresh token lifetime in seconds (default 30 days)
        """
        super().__init__(base_url=base_url)
        self._password_hash = hashlib.sha256(password.encode()).hexdigest()
        self._token_lifetime = token_lifetime
        self._refresh_token_lifetime = refresh_token_lifetime

        # In-memory storage
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._auth_codes: dict[str, AuthorizationCode] = {}
        self._access_tokens: dict[str, AccessToken] = {}
        self._refresh_tokens: dict[str, RefreshToken] = {}

    def _generate_token(self, prefix: str = "") -> str:
        """Generate a secure random token."""
        return f"{prefix}{secrets.token_urlsafe(32)}"

    def _verify_password(self, password: str) -> bool:
        """Verify password using constant-time comparison."""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return hmac.compare_digest(password_hash, self._password_hash)

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        """Get registered client by ID."""
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        """Register a new client (auto-approve for single user)."""
        self._clients[client_info.client_id] = client_info

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        """Handle authorization request.

        For single-user, we serve a simple HTML login form.
        """
        # Build the login form URL
        form_params = {
            "client_id": client.client_id,
            "redirect_uri": str(params.redirect_uri),
            "state": params.state or "",
            "code_challenge": params.code_challenge,
            "scopes": ",".join(params.scopes or []),
        }
        if params.resource:
            form_params["resource"] = params.resource

        # Return URL to our login endpoint
        login_url = f"{self.base_url}/oauth/login?{urlencode(form_params)}"
        return login_url

    def create_authorization_code(
        self,
        client_id: str,
        redirect_uri: str,
        code_challenge: str,
        scopes: list[str],
        resource: str | None = None,
    ) -> str:
        """Create and store an authorization code."""
        code = self._generate_token("code_")
        auth_code = AuthorizationCode(
            code=code,
            client_id=client_id,
            redirect_uri=AnyHttpUrl(redirect_uri),
            redirect_uri_provided_explicitly=True,
            code_challenge=code_challenge,
            scopes=scopes,
            expires_at=time.time() + 600,  # 10 minutes
            resource=resource,
        )
        self._auth_codes[code] = auth_code
        return code

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        """Load authorization code."""
        code = self._auth_codes.get(authorization_code)
        if code and code.expires_at > time.time() and code.client_id == client.client_id:
            return code
        return None

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        """Exchange authorization code for tokens."""
        # Remove used code
        self._auth_codes.pop(authorization_code.code, None)

        # Create access token
        access_token = self._generate_token("at_")
        expires_at = int(time.time()) + self._token_lifetime
        self._access_tokens[access_token] = AccessToken(
            token=access_token,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=expires_at,
            resource=authorization_code.resource,
        )

        # Create refresh token
        refresh_token = self._generate_token("rt_")
        self._refresh_tokens[refresh_token] = RefreshToken(
            token=refresh_token,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=int(time.time()) + self._refresh_token_lifetime,
        )

        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=self._token_lifetime,
            refresh_token=refresh_token,
            scope=" ".join(authorization_code.scopes) if authorization_code.scopes else None,
        )

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        """Load refresh token."""
        token = self._refresh_tokens.get(refresh_token)
        if token and token.client_id == client.client_id:
            if token.expires_at is None or token.expires_at > time.time():
                return token
        return None

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        """Exchange refresh token for new tokens."""
        # Remove old refresh token
        self._refresh_tokens.pop(refresh_token.token, None)

        # Use requested scopes or original scopes
        token_scopes = scopes if scopes else refresh_token.scopes

        # Create new access token
        access_token = self._generate_token("at_")
        expires_at = int(time.time()) + self._token_lifetime
        self._access_tokens[access_token] = AccessToken(
            token=access_token,
            client_id=client.client_id,
            scopes=token_scopes,
            expires_at=expires_at,
        )

        # Create new refresh token
        new_refresh_token = self._generate_token("rt_")
        self._refresh_tokens[new_refresh_token] = RefreshToken(
            token=new_refresh_token,
            client_id=client.client_id,
            scopes=token_scopes,
            expires_at=int(time.time()) + self._refresh_token_lifetime,
        )

        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=self._token_lifetime,
            refresh_token=new_refresh_token,
            scope=" ".join(token_scopes) if token_scopes else None,
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        """Verify and load access token."""
        access_token = self._access_tokens.get(token)
        if access_token:
            if access_token.expires_at is None or access_token.expires_at > time.time():
                return access_token
            # Expired, remove it
            self._access_tokens.pop(token, None)
        return None

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        """Revoke a token."""
        if isinstance(token, AccessToken):
            self._access_tokens.pop(token.token, None)
        elif isinstance(token, RefreshToken):
            self._refresh_tokens.pop(token.token, None)

    def verify_password_and_create_code(
        self,
        password: str,
        client_id: str,
        redirect_uri: str,
        code_challenge: str,
        scopes: list[str],
        resource: str | None = None,
    ) -> str | None:
        """Verify password and create authorization code if valid.

        Returns authorization code if password is correct, None otherwise.
        """
        if self._verify_password(password):
            return self.create_authorization_code(
                client_id=client_id,
                redirect_uri=redirect_uri,
                code_challenge=code_challenge,
                scopes=scopes,
                resource=resource,
            )
        return None
