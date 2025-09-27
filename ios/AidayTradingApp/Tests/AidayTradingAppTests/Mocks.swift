import Foundation
@testable import AidayTradingApp

final class MockAuthService: AuthServiceProtocol {
    var loginResult: Result<UserSessionContext, Error> = .failure(MockError.notConfigured)
    var signupResult: Result<UserProfile, Error> = .failure(MockError.notConfigured)
    var refreshResult: Result<AuthTokens, Error> = .failure(MockError.notConfigured)
    var profileResult: Result<UserProfile, Error> = .failure(MockError.notConfigured)
    var logoutCalled = false
    var passwordResetEmails: [String] = []

    func signup(username: String, email: String, password: String) async throws -> UserProfile {
        try await signupResult.get()
    }

    func login(username: String, password: String) async throws -> UserSessionContext {
        try await loginResult.get()
    }

    func refresh(using refreshToken: String) async throws -> AuthTokens {
        try await refreshResult.get()
    }

    func loadProfile(accessToken: String) async throws -> UserProfile {
        try await profileResult.get()
    }

    func logout(accessToken: String) async throws {
        logoutCalled = true
    }

    func requestPasswordReset(email: String) async throws {
        passwordResetEmails.append(email)
    }
}

final class MockTokenStorage: SecureTokenStorage {
    var storedTokens: AuthTokens?
    var saveCalled = false
    var deleteCalled = false

    func save(tokens: AuthTokens) throws {
        storedTokens = tokens
        saveCalled = true
    }

    func load() throws -> AuthTokens? {
        storedTokens
    }

    func delete() throws {
        storedTokens = nil
        deleteCalled = true
    }
}

struct MockBiometricAuthenticator: BiometricAuthenticating {
    var shouldFail = false

    func authenticate(reason: String) async throws {
        if shouldFail {
            throw MockError.biometricFailed
        }
    }
}

final class MockIdleTimeoutManager: IdleTimeoutHandling {
    var onTimeout: (() -> Void)?
    private(set) var startCalled = false
    private(set) var stopCalled = false
    private(set) var resetCalled = false

    func start() {
        startCalled = true
    }

    func reset() {
        resetCalled = true
    }

    func stop() {
        stopCalled = true
    }
}

enum MockError: Error {
    case notConfigured
    case biometricFailed
}

final class MockApprovalService: ApprovalServiceProtocol {
    var statusResult: Result<UserProfile.ApprovalStatus, Error> = .failure(MockError.notConfigured)

    func fetchStatus(username: String, email: String) async throws -> UserProfile.ApprovalStatus {
        try await statusResult.get()
    }
}

final class MockLocalNotificationScheduler: LocalNotificationScheduling {
    private(set) var scheduleCallCount = 0
    private(set) var cancelCallCount = 0

    func scheduleRealtimeStallNotification() async {
        scheduleCallCount += 1
    }

    func cancelRealtimeStallNotification() {
        cancelCallCount += 1
    }
}

final class MockPushNotificationService: PushNotificationRegistering, NotificationPreferencesServicing {
    private(set) var lastToken: String?
    private(set) var registerCallCount = 0
    private(set) var updateCallCount = 0
    private(set) var unregisterCallCount = 0
    var preferences: NotificationPreferences = .default

    func register(deviceToken: String, accessToken: String) async throws {
        lastToken = deviceToken
        registerCallCount += 1
    }

    func fetchPreferences(accessToken: String) async throws -> NotificationPreferences {
        preferences
    }

    func update(preferences: NotificationPreferences, accessToken: String) async throws -> NotificationPreferences {
        updateCallCount += 1
        self.preferences = preferences
        return preferences
    }

    func unregister(deviceToken: String, accessToken: String) async throws {
        unregisterCallCount += 1
    }
}

final class MockRealtimeService: RealtimeServiceProtocol {
    weak var delegate: RealtimeServiceDelegate?
    private(set) var connectCallCount = 0
    private(set) var disconnectCallCount = 0

    func connect(accessToken: String) {
        connectCallCount += 1
    }

    func disconnect() {
        disconnectCallCount += 1
    }
}

final class MockReportingService: ReportingServiceProtocol {
    var statusResult: Result<SystemStatus, Error> = .failure(MockError.notConfigured)
    var profitResult: Result<ProfitSummary, Error> = .failure(MockError.notConfigured)
    var balanceResult: Result<Decimal, Error> = .failure(MockError.notConfigured)
    var equityResult: Result<[EquityCurvePoint], Error> = .failure(MockError.notConfigured)
    var tradesResult: Result<TradesPage, Error> = .failure(MockError.notConfigured)

    func fetchSystemStatus(accessToken: String) async throws -> SystemStatus {
        try statusResult.get()
    }

    func fetchProfitSummary(accessToken: String) async throws -> ProfitSummary {
        try profitResult.get()
    }

    func fetchCurrentBalance(accessToken: String) async throws -> Decimal {
        try balanceResult.get()
    }

    func fetchEquityCurve(accessToken: String, start: Date?, end: Date?, limit: Int?) async throws -> [EquityCurvePoint] {
        try equityResult.get()
    }

    func fetchTrades(
        accessToken: String,
        page: Int,
        pageSize: Int,
        symbol: String?,
        side: TradeSide?,
        start: Date?,
        end: Date?
    ) async throws -> TradesPage {
        try tradesResult.get()
    }
}
