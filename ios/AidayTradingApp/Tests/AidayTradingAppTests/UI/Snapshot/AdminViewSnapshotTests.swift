import SwiftUI
import XCTest
@testable import AidayTradingApp

@MainActor
final class AdminViewSnapshotTests: XCTestCase {
    func testAdminViewRendersWithSampleData() async throws {
        let repository = SnapshotAdminRepository()
        let changeLog = SnapshotChangeLogRepository()
        let viewModel = AdminViewModel(actor: "admin", repository: repository, changeLogRepository: changeLog)
        await viewModel.riskViewModel.load()
        await viewModel.userManagementViewModel.loadUsers()
        await viewModel.botControlViewModel.loadStatus()
        let notificationManager = NotificationManager(
            pushService: MockPushNotificationService(),
            localScheduler: MockLocalNotificationScheduler(),
            persistence: NotificationPersistenceController(inMemory: true)
        )
        viewModel.attach(notificationManager: notificationManager)
        let view = AdminView(viewModel: viewModel)
            .environmentObject(notificationManager)
        let renderer = ImageRenderer(content: view.frame(width: 390, height: 844))
        #if canImport(UIKit)
        XCTAssertNotNil(renderer.uiImage)
        #elseif canImport(AppKit)
        XCTAssertNotNil(renderer.nsImage)
        #else
        throw XCTSkip("Snapshot rendering not supported on this platform")
        #endif
    }

    func testUserDetailSnapshot() async throws {
        let repository = MockAdminRepository()
        let user = AdminUser(
            id: UUID(),
            username: "snapshot",
            email: "snapshot@example.com",
            role: .viewer,
            status: .pending
        )
        repository.users = [user]
        let viewModel = UserManagementViewModel(repository: repository)
        await viewModel.loadUsers()
        let detail = UserDetailView(viewModel: viewModel, user: user) {}
        let renderer = ImageRenderer(content: detail.frame(width: 390, height: 600))
        #if canImport(UIKit)
        XCTAssertNotNil(renderer.uiImage)
        #elseif canImport(AppKit)
        XCTAssertNotNil(renderer.nsImage)
        #else
        throw XCTSkip("Snapshot rendering not supported on this platform")
        #endif
    }
}

private final class SnapshotAdminRepository: AdminRepository {
    private var risk = RiskConfiguration(
        maxDrawdownPercent: 22,
        dailyLossLimitPercent: 6,
        riskPerTrade: 0.03,
        maxOpenPositions: 4,
        atrStopLossMultiplier: 1.8,
        atrTakeProfitMultiplier: 2.4
    )

    private var users: [AdminUser] = [
        AdminUser(id: UUID(), username: "bot-ops", email: "bot.ops@example.com", role: .admin, status: .active),
        AdminUser(id: UUID(), username: "analyst", email: "analyst@example.com", role: .viewer, status: .pending)
    ]

    private var status = BotStatus(running: true, mode: .paper, lastUpdated: Date())

    func fetchRiskConfiguration() async throws -> RiskConfiguration { risk }

    func updateRiskConfiguration(_ configuration: RiskConfiguration) async throws -> RiskConfiguration {
        risk = configuration
        return configuration
    }

    func fetchUsers() async throws -> [AdminUser] { users }

    func updateUser(id: UUID, role: AdminUser.Role?, status: AdminUser.Status?) async throws -> AdminUser {
        guard let index = users.firstIndex(where: { $0.id == id }) else {
            throw MockError.notConfigured
        }
        var updated = users[index]
        if let role { updated.role = role }
        if let status { updated.status = status }
        users[index] = updated
        return updated
    }

    func resetPassword(id: UUID) async throws {}

    func fetchBotStatus() async throws -> BotStatus { status }

    func startBot() async throws -> BotStatus {
        status.running = true
        return status
    }

    func stopBot() async throws -> BotStatus {
        status.running = false
        return status
    }

    func setBotMode(_ mode: BotMode) async throws -> BotStatus {
        status.mode = mode
        return status
    }
}

private final class SnapshotChangeLogRepository: AdminChangeLogRepositoryProtocol {
    private var entries: [AdminChangeLogEntry] = [
        AdminChangeLogEntry(
            id: UUID(),
            timestamp: Date().addingTimeInterval(-3600),
            actor: "admin",
            summary: "Risk guardrails updated",
            details: "Max DD 22%, Risk/trade 3%",
            category: .risk
        ),
        AdminChangeLogEntry(
            id: UUID(),
            timestamp: Date().addingTimeInterval(-1800),
            actor: "admin",
            summary: "Bot switched to paper",
            details: "Mode: Paper, Running: Yes",
            category: .bot
        )
    ]

    func record(_ entry: AdminChangeLogEntry) throws {
        entries.append(entry)
    }

    func fetchLatest(limit: Int) throws -> [AdminChangeLogEntry] {
        Array(entries.prefix(limit))
    }
}
