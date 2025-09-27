import Combine
import XCTest
@testable import AidayTradingApp

@MainActor
final class NotificationManagerTests: XCTestCase {
    func testRegistersPushTokenWhenSessionAvailable() async throws {
        let pushService = MockPushNotificationService()
        let persistence = NotificationPersistenceController(inMemory: true)
        let manager = NotificationManager(pushService: pushService, localScheduler: MockLocalNotificationScheduler(), persistence: persistence)
        let profile = UserProfile(
            id: UUID(),
            username: "Trader",
            email: "trader@example.com",
            role: .viewer,
            approvalStatus: .approved
        )
        let tokens = AuthTokens(accessToken: "access", refreshToken: "refresh", accessTokenExpiry: Date().addingTimeInterval(3600))

        manager.updateSessionState(.authenticated(UserSessionContext(profile: profile, tokens: tokens)))
        manager.handleFCMToken("abc123")
        try await Task.sleep(nanoseconds: 50_000_000)

        XCTAssertEqual(pushService.lastToken, "abc123")
        XCTAssertEqual(pushService.registerCallCount, 1)

        manager.handleFCMToken("abc123")
        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(pushService.registerCallCount, 1, "Duplicate tokens should not trigger re-registration")
    }

    func testRoutesNotificationsToCorrectTab() async throws {
        let manager = NotificationManager(pushService: MockPushNotificationService(), localScheduler: MockLocalNotificationScheduler(), persistence: NotificationPersistenceController(inMemory: true))
        manager.handleRemoteNotification(userInfo: ["target": "trades", "aps": ["alert": ["title": "Test"]]])
        XCTAssertEqual(manager.consumePendingTab(), .trades)

        manager.handleRemoteNotification(userInfo: ["target": "notifications", "aps": ["alert": ["title": "Test"]]])
        XCTAssertEqual(manager.consumePendingTab(), .notifications)

        manager.handleRemoteNotification(userInfo: ["target": "unknown", "aps": ["alert": ["title": "Test"]]])
        XCTAssertNil(manager.consumePendingTab())
    }

    func testAdminCanUpdatePreferences() async throws {
        let pushService = MockPushNotificationService()
        let persistence = NotificationPersistenceController(inMemory: true)
        let manager = NotificationManager(pushService: pushService, localScheduler: MockLocalNotificationScheduler(), persistence: persistence)
        let profile = UserProfile(
            id: UUID(),
            username: "Admin",
            email: "admin@example.com",
            role: .admin,
            approvalStatus: .approved
        )
        let tokens = AuthTokens(accessToken: "access", refreshToken: "refresh", accessTokenExpiry: Date().addingTimeInterval(3600))

        manager.updateSessionState(.authenticated(UserSessionContext(profile: profile, tokens: tokens)))
        manager.setBotEvents(enabled: false)
        try await Task.sleep(nanoseconds: 50_000_000)

        XCTAssertEqual(pushService.updateCallCount, 1)
        XCTAssertFalse(manager.preferences.botEventsEnabled)
    }

    func testViewerCannotModifyPreferences() async throws {
        let pushService = MockPushNotificationService()
        let manager = NotificationManager(pushService: pushService, localScheduler: MockLocalNotificationScheduler(), persistence: NotificationPersistenceController(inMemory: true))
        let profile = UserProfile(
            id: UUID(),
            username: "Viewer",
            email: "viewer@example.com",
            role: .viewer,
            approvalStatus: .approved
        )
        let tokens = AuthTokens(accessToken: "access", refreshToken: "refresh", accessTokenExpiry: Date().addingTimeInterval(3600))
        manager.updateSessionState(.authenticated(UserSessionContext(profile: profile, tokens: tokens)))

        manager.setBotEvents(enabled: false)
        try await Task.sleep(nanoseconds: 20_000_000)

        XCTAssertEqual(pushService.updateCallCount, 0)
        XCTAssertTrue(manager.preferences.botEventsEnabled)
    }

    func testRealtimeTradeStoresNotification() async throws {
        let pushService = MockPushNotificationService()
        let persistence = NotificationPersistenceController(inMemory: true)
        let manager = NotificationManager(pushService: pushService, localScheduler: MockLocalNotificationScheduler(), persistence: persistence)
        let client = MockWebSocketClient()
        manager.bind(to: client)

        let trade = Trade(
            id: "t1",
            symbol: "BTC/USD",
            side: "buy",
            quantity: 0.01,
            price: 20_000,
            pnl: 120,
            pnlPercent: 0,
            openedAt: Date(),
            closedAt: Date(),
            timeZone: AppConfig.centralTimeZone
        )
        client.tradePublisher.send(trade)
        try await Task.sleep(nanoseconds: 20_000_000)

        XCTAssertFalse(manager.notifications.isEmpty)
        XCTAssertEqual(manager.notifications.first?.kind, .system)
    }
}

private final class MockWebSocketClient: TradingWebSocketClientProtocol {
    let equityPublisher = PassthroughSubject<EquityPoint, Never>()
    let tradePublisher = PassthroughSubject<Trade, Never>()
    let statusPublisher = PassthroughSubject<Status, Never>()
    let connectionState = CurrentValueSubject<TradingWebSocketClient.ConnectionState, Never>(.disconnected)

    func connect(accessToken: String) {}
    func disconnect() {}
    func sendPing() {}
}
