import SwiftUI
import XCTest
@testable import AidayTradingApp

#if canImport(SwiftUI)

@MainActor
final class HomeViewSnapshotTests: XCTestCase {
    func testHomeViewRendersWithSampleData() throws {
        let repository = StubDashboardRepository()
        let viewModel = HomeViewModel(repository: repository)
        viewModel.status = repository.sampleStatus
        viewModel.profit = repository.sampleProfit
        viewModel.equity = repository.sampleEquity
        viewModel.lastUpdated = Date()
        viewModel.isLoading = false

        let view = HomeView(viewModel: viewModel)
        let renderer = ImageRenderer(content: view.frame(width: 390, height: 844))

        #if canImport(UIKit)
        XCTAssertNotNil(renderer.uiImage)
        #elseif canImport(AppKit)
        XCTAssertNotNil(renderer.nsImage)
        #else
        throw XCTSkip("Snapshot rendering not supported on this platform")
        #endif
    }
}

private final class StubDashboardRepository: DashboardRepository {
    let sampleStatus = Status(running: true, uptimeSeconds: 7200)
    let sampleProfit = ProfitSnapshot(balance: 12_500, pnlAbsolute: 450, pnlPercent: 3.5, winRate: 0.62)
    let sampleEquity = [
        EquityPoint(timestamp: Date(timeIntervalSince1970: 1_696_000_000), equity: 11_000),
        EquityPoint(timestamp: Date(timeIntervalSince1970: 1_696_006_000), equity: 11_500),
        EquityPoint(timestamp: Date(timeIntervalSince1970: 1_696_012_000), equity: 12_500)
    ]

    func fetchStatus() async throws -> Status { sampleStatus }
    func fetchProfit() async throws -> ProfitSnapshot { sampleProfit }
    func fetchEquityCurve() async throws -> [EquityPoint] { sampleEquity }
    func cachedStatus() -> Status? { sampleStatus }
    func cachedProfit() -> ProfitSnapshot? { sampleProfit }
    func cachedEquityCurve() -> [EquityPoint]? { sampleEquity }
}

#endif
