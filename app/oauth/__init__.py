"""OAuth 2.1 + Dynamic Client Registration для multi-user MCP.

Дизайн:
- public clients (без secret), PKCE S256 обязателен.
- shared password авторизация: все авторизованные коллеги вводят один пароль на /authorize.
- access_token — JWT HS256, подписан MCP_SECRET_KEY.
- refresh_token — opaque, in-memory store с TTL.
- DCR clients — in-memory store. После рестарта клиент перерегистрируется автоматически.
"""
