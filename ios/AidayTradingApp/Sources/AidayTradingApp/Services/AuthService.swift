import Foundation

protocol AuthServiceProtocol {
    func signup(username: String, email: String, password: String) async throws -> UserProfile
    func login(email: String, password: String) async throws -> UserSessionContext
    func refresh(using refreshToken: String) async throws -> AuthTokens
    func loadProfile(accessToken: String) async throws -> UserProfile
}

struct AuthService: AuthServiceProtocol {
    private let apiClient: APIClientProtocol

    init(apiClient: APIClientProtocol = APIClient()) {
        self.apiClient = apiClient
    }

    func signup(username: String, email: String, password: String) async throws -> UserProfile {
        let response: AuthResponse = try await apiClient.send(AuthRequest.signup(username: username, email: email, password: password), decode: AuthResponse.self)
        return response.user
    }

    func login(email: String, password: String) async throws -> UserSessionContext {
        let response: AuthResponse = try await apiClient.send(AuthRequest.login(email: email, password: password), decode: AuthResponse.self)
        return UserSessionContext(profile: response.user, tokens: response.tokens)
    }

    func refresh(using refreshToken: String) async throws -> AuthTokens {
        let response: AuthResponse = try await apiClient.send(AuthRequest.refresh(refreshToken: refreshToken), decode: AuthResponse.self)
        return response.tokens
    }

    func loadProfile(accessToken: String) async throws -> UserProfile {
        try await apiClient.send(AuthRequest.profile(accessToken: accessToken), decode: UserProfile.self)
    }
}
