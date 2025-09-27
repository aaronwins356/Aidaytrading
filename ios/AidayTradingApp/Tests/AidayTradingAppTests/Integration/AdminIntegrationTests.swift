import XCTest
@testable import AidayTradingApp

@MainActor
final class AdminIntegrationTests: XCTestCase {
    func testRiskUpdateRecordsChangeLog() async {
        let repository = MockAdminRepository()
        let changeLog = MockAdminChangeLogRepository()
        let viewModel = AdminViewModel(actor: "admin", repository: repository, changeLogRepository: changeLog)
        viewModel.riskViewModel.configuration.maxDrawdownPercent = 30
        await viewModel.riskViewModel.save()
        XCTAssertEqual(changeLog.entries.count, 1)
        XCTAssertEqual(viewModel.changeLog.first?.category, .risk)
    }

    func testUserApprovalCreatesChangeLogEntry() async {
        let repository = MockAdminRepository()
        let changeLog = MockAdminChangeLogRepository()
        let user = AdminUser(
            id: UUID(),
            username: "maria",
            email: "maria@example.com",
            role: .viewer,
            status: .pending
        )
        repository.users = [user]
        let viewModel = AdminViewModel(actor: "admin", repository: repository, changeLogRepository: changeLog)
        await viewModel.userManagementViewModel.loadUsers()
        await viewModel.userManagementViewModel.approve(user)
        XCTAssertEqual(repository.users.first?.status, .active)
        XCTAssertEqual(changeLog.entries.last?.category, .user)
    }

    func testBotStopRecordsChange() async {
        let repository = MockAdminRepository()
        repository.botStatus = BotStatus(running: true, mode: .live, lastUpdated: Date())
        let changeLog = MockAdminChangeLogRepository()
        let viewModel = AdminViewModel(actor: "admin", repository: repository, changeLogRepository: changeLog)
        await viewModel.botControlViewModel.loadStatus()
        await viewModel.botControlViewModel.stopBot()
        XCTAssertFalse(repository.botStatus.running)
        XCTAssertEqual(changeLog.entries.last?.category, .bot)
    }
}
