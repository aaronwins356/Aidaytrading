import Foundation
@testable import AidayTradingApp

final class MockAuthService: AuthServiceProtocol {
    var loginResult: Result<UserSessionContext, Error> = .failure(MockError.notConfigured)
    var signupResult: Result<UserProfile, Error> = .failure(MockError.notConfigured)
    var refreshResult: Result<AuthTokens, Error> = .failure(MockError.notConfigured)
    var profileResult: Result<UserProfile, Error> = .failure(MockError.notConfigured)

    func signup(username: String, email: String, password: String) async throws -> UserProfile {
        try await signupResult.get()
    }

    func login(email: String, password: String) async throws -> UserSessionContext {
        try await loginResult.get()
    }

    func refresh(using refreshToken: String) async throws -> AuthTokens {
        try await refreshResult.get()
    }

    func loadProfile(accessToken: String) async throws -> UserProfile {
        try await profileResult.get()
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
