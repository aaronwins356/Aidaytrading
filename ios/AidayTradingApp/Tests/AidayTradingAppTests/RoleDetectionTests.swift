import XCTest
@testable import AidayTradingApp

final class RoleDetectionTests: XCTestCase {
    func testViewerSeesThreeTabs() {
        let profile = UserProfile(id: UUID(), username: "Viewer", email: "viewer@example.com", role: .viewer, approvalStatus: .approved)
        let tokens = AuthTokens(accessToken: "token", refreshToken: "refresh", accessTokenExpiry: .distantFuture)
        let context = UserSessionContext(profile: profile, tokens: tokens)
        XCTAssertEqual(RoleManager.accessibleTabs(for: profile), [.home, .calendar, .trades])
        XCTAssertFalse(RoleManager.canApproveUsers(profile))
        let view = MainTabView(context: context)
        XCTAssertNotNil(view)
    }

    func testAdminSeesAdminTab() {
        let profile = UserProfile(id: UUID(), username: "Admin", email: "admin@example.com", role: .admin, approvalStatus: .approved)
        let tokens = AuthTokens(accessToken: "token", refreshToken: "refresh", accessTokenExpiry: .distantFuture)
        let context = UserSessionContext(profile: profile, tokens: tokens)
        XCTAssertEqual(RoleManager.accessibleTabs(for: profile), [.home, .calendar, .trades, .admin])
        XCTAssertTrue(RoleManager.canApproveUsers(profile))
        XCTAssertNotNil(MainTabView(context: context))
    }
}
