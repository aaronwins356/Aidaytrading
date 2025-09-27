import XCTest
import SwiftUI
@testable import AidayTradingApp

@MainActor
final class AdminViewModelsTests: XCTestCase {
    func testRiskViewModelClampsValues() {
        let repository = MockAdminRepository()
        let viewModel = RiskViewModel(repository: repository)
        let binding = viewModel.binding(
            for: \.maxDrawdownPercent,
            range: RiskConfiguration.ranges.maxDrawdownPercent
        )
        binding.wrappedValue = 120
        XCTAssertEqual(viewModel.configuration.maxDrawdownPercent, RiskConfiguration.ranges.maxDrawdownPercent.upperBound)
    }

    func testRiskSavePersistsConfigurationAndRecordsChange() async {
        let repository = MockAdminRepository()
        let recorder = SpyActionRecorder()
        let viewModel = RiskViewModel(repository: repository)
        viewModel.actionRecorder = recorder
        viewModel.configuration.maxDrawdownPercent = 25
        await viewModel.save()
        XCTAssertEqual(repository.riskConfiguration.maxDrawdownPercent, 25)
        XCTAssertFalse(recorder.recorded.isEmpty)
    }

    func testUserApprovalFlow() async {
        let repository = MockAdminRepository()
        let pendingUser = AdminUser(
            id: UUID(),
            username: "alice",
            email: "alice@example.com",
            role: .viewer,
            status: .pending
        )
        repository.users = [pendingUser]
        let viewModel = UserManagementViewModel(repository: repository)
        await viewModel.loadUsers()
        await viewModel.approve(pendingUser)
        XCTAssertEqual(viewModel.users.first?.status, .active)
        XCTAssertEqual(repository.users.first?.status, .active)
    }

    func testBotControlModeSwitches() async {
        let repository = MockAdminRepository()
        repository.botStatus = BotStatus(running: false, mode: .paper, lastUpdated: Date())
        let recorder = SpyActionRecorder()
        let viewModel = BotControlViewModel(repository: repository)
        viewModel.actionRecorder = recorder
        await viewModel.loadStatus()
        await viewModel.startBot()
        XCTAssertTrue(repository.botStatus.running)
        await viewModel.setMode(.live)
        XCTAssertEqual(repository.botStatus.mode, .live)
        XCTAssertFalse(recorder.recorded.isEmpty)
    }
}
