import Foundation

actor AuthInterceptor {
    private let tokenStorage: SecureTokenStorage
    private let authService: AuthServiceProtocol
    private var cachedTokens: AuthTokens?

    init(tokenStorage: SecureTokenStorage = KeychainStorage(), authService: AuthServiceProtocol = AuthService()) {
        self.tokenStorage = tokenStorage
        self.authService = authService
        self.cachedTokens = try? tokenStorage.load()
    }

    func authorize(_ request: URLRequest) async throws -> URLRequest {
        var request = request
        let tokens = try await ensureValidTokens()
        request.setValue("Bearer \(tokens.accessToken)", forHTTPHeaderField: "Authorization")
        return request
    }

    func refreshAfterUnauthorized() async throws -> Bool {
        guard let tokens = try await loadTokens() else {
            return false
        }
        let refreshed = try await authService.refresh(using: tokens.refreshToken)
        try tokenStorage.save(tokens: refreshed)
        cachedTokens = refreshed
        return true
    }

    private func ensureValidTokens() async throws -> AuthTokens {
        if let cached = cachedTokens, cached.isAccessTokenValid {
            return cached
        }
        guard var stored = try tokenStorage.load() else {
            throw SecretsError.missingToken
        }
        if !stored.isAccessTokenValid {
            stored = try await authService.refresh(using: stored.refreshToken)
            try tokenStorage.save(tokens: stored)
        }
        cachedTokens = stored
        return stored
    }

    private func loadTokens() throws -> AuthTokens? {
        if let cached = cachedTokens {
            return cached
        }
        let stored = try tokenStorage.load()
        cachedTokens = stored
        return stored
    }
}

extension AuthInterceptor: AuthIntercepting {}
