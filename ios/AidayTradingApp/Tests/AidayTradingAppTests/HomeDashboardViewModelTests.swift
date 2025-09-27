import XCTest
@testable import AidayTradingApp

@MainActor
final class HomeDashboardViewModelTests: XCTestCase {
    private let tokens = AuthTokens(accessToken: "token", refreshToken: "refresh", accessTokenExpiry: .distantFuture)

    func testLoadDataPopulatesMetricsAndFallbacksBalance() async {
        let service = MockReportingService()
        service.statusResult = .success(SystemStatus(running: true, uptimeSeconds: 300))
        let profit = ProfitSummary(
            currentBalance: Decimal(string: "10000")!,
            totalPLAmount: Decimal(string: "250")!,
            totalPLPercent: Decimal(string: "2.5")!,
            winRate: 0.55
        )
        service.profitResult = .success(profit)
        service.balanceResult = .failure(MockError.notConfigured)
        service.equityResult = .success([
            EquityCurvePoint(timestamp: Date(), equity: Decimal(string: "10000")!)
        ])

        let viewModel = HomeDashboardViewModel(
            accessToken: tokens.accessToken,
            reportingService: service,
            realtimeService: MockRealtimeService(),
            notificationScheduler: MockLocalNotificationScheduler()
        )
        await viewModel.loadData()

        XCTAssertEqual(viewModel.profitSummary?.currentBalance, profit.currentBalance)
        XCTAssertEqual(viewModel.balance, profit.currentBalance)
        XCTAssertEqual(viewModel.systemStatus?.running, true)
        XCTAssertFalse(viewModel.equitySeries.isEmpty)
    }
}
