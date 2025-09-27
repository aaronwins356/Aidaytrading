import XCTest
@testable import AidayTradingApp

final class LoginFlowTests: XCTestCase {
    func testSuccessfulLoginStoresTokensAndAuthenticates() async throws {
        let tokens = AuthTokens(accessToken: "access", refreshToken: "refresh", accessTokenExpiry: .distantFuture)
        let profile = UserProfile(id: UUID(), username: "Trader", email: "user@example.com", role: .viewer, approvalStatus: .approved)
        let context = UserSessionContext(profile: profile, tokens: tokens)

        let authService = MockAuthService()
        authService.loginResult = .success(context)
        let tokenStorage = MockTokenStorage()
        let idleManager = MockIdleTimeoutManager()
        let session = SessionStore(
            authService: authService,
            tokenStorage: tokenStorage,
            biometricAuthenticator: MockBiometricAuthenticator(),
            idleManager: idleManager,
            previewState: .loggedOut
        )

        await session.login(email: profile.email, password: "Secure123")

        XCTAssertTrue(tokenStorage.saveCalled)
        guard case let .authenticated(sessionContext) = session.state else {
            return XCTFail("Expected authenticated state")
        }
        XCTAssertEqual(sessionContext.profile.role, .viewer)
        XCTAssertTrue(idleManager.startCalled)
    }

    func testLoginFailureClearsTokens() async {
        let authService = MockAuthService()
        authService.loginResult = .failure(APIErrorResponse(message: "Invalid credentials"))
        let tokenStorage = MockTokenStorage()
        tokenStorage.storedTokens = AuthTokens(accessToken: "stale", refreshToken: "refresh", accessTokenExpiry: .distantPast)
        let session = SessionStore(
            authService: authService,
            tokenStorage: tokenStorage,
            biometricAuthenticator: MockBiometricAuthenticator(),
            idleManager: MockIdleTimeoutManager(),
            previewState: .loggedOut
        )

        await session.login(email: "user@example.com", password: "bad")

        XCTAssertTrue(tokenStorage.deleteCalled)
        guard case .loggedOut = session.state else {
            return XCTFail("Expected logged out state")
        }
    }
}
