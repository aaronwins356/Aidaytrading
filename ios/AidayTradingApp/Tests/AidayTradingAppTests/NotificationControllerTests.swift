import XCTest
@testable import AidayTradingApp

final class NotificationControllerTests: XCTestCase {
    func testRegistersPushTokenWhenSessionAvailable() async {
        let pushService = MockPushNotificationService()
        let controller = NotificationController(pushService: pushService, localScheduler: MockLocalNotificationScheduler())
        let profile = UserProfile(
            id: UUID(),
            username: "Trader",
            email: "trader@example.com",
            role: .viewer,
            approvalStatus: .approved
        )
        let tokens = AuthTokens(accessToken: "access", refreshToken: "refresh", accessTokenExpiry: Date().addingTimeInterval(3600))
        controller.updateSessionState(.authenticated(UserSessionContext(profile: profile, tokens: tokens)))

        controller.registerFCMToken("abc123")
        try? await Task.sleep(nanoseconds: 50_000_000)

        XCTAssertEqual(pushService.lastToken, "abc123")
        XCTAssertEqual(pushService.registerCallCount, 1)

        controller.registerFCMToken("abc123")
        try? await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(pushService.registerCallCount, 1, "Duplicate tokens should not trigger another registration")
    }

    func testRoutesNotificationsToCorrectTab() {
        let controller = NotificationController(pushService: MockPushNotificationService(), localScheduler: MockLocalNotificationScheduler())
        controller.handleRemoteNotification(userInfo: ["target": "trades"])
        XCTAssertEqual(controller.consumePendingTab(), .trades)

        controller.handleRemoteNotification(userInfo: ["target": "calendar"])
        XCTAssertEqual(controller.consumePendingTab(), .calendar)

        controller.handleRemoteNotification(userInfo: ["target": "unknown"])
        XCTAssertEqual(controller.consumePendingTab(), .home)
    }
}
