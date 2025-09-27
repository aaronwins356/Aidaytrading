import XCTest
@testable import AidayTradingApp

final class SignupApprovalIntegrationTests: XCTestCase {
    func testSignupLeadsToPendingStateUntilApproved() async {
        let pendingProfile = UserProfile(id: UUID(), username: "NewUser", email: "new@example.com", role: .viewer, approvalStatus: .pending)
        let approvedProfile = UserProfile(id: pendingProfile.id, username: pendingProfile.username, email: pendingProfile.email, role: .viewer, approvalStatus: .approved)
        let tokens = AuthTokens(accessToken: "access", refreshToken: "refresh", accessTokenExpiry: .distantFuture)
        let context = UserSessionContext(profile: approvedProfile, tokens: tokens)

        let authService = MockAuthService()
        authService.signupResult = .success(pendingProfile)
        authService.loginResult = .success(context)
        authService.profileResult = .success(approvedProfile)

        let session = SessionStore(
            authService: authService,
            tokenStorage: MockTokenStorage(),
            biometricAuthenticator: MockBiometricAuthenticator(),
            idleManager: MockIdleTimeoutManager(),
            approvalService: MockApprovalService(),
            previewState: .loggedOut
        )

        await session.signup(username: "NewUser", email: pendingProfile.email, password: "Password1")
        guard case let .pendingApproval(pendingContext) = session.state else {
            return XCTFail("Expected pending approval state")
        }
        XCTAssertEqual(pendingContext.email, pendingProfile.email)

        await session.login(username: approvedProfile.username, password: "Password1")
        guard case let .authenticated(authenticatedContext) = session.state else {
            return XCTFail("Expected authenticated state")
        }
        XCTAssertEqual(authenticatedContext.profile.id, approvedProfile.id)
    }
}
