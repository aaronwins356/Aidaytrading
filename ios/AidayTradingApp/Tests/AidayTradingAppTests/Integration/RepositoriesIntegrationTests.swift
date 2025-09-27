import XCTest
@testable import AidayTradingApp

final class RepositoriesIntegrationTests: XCTestCase {
    override func setUp() {
        super.setUp()
        DiskCache.shared.removeValue(for: "profit_latest.json")
        DiskCache.shared.removeValue(for: "equity_latest.json")
        DiskCache.shared.removeValue(for: "status_latest.json")
        DiskCache.shared.removeValue(for: "trades_page_1.json")
    }

    func testHomeViewModelBindsRepositoryData() async throws {
        let client = MockAPIClient()
        let statusJSON: [String: Any] = ["running": true, "uptime_seconds": 3600]
        client.responses["/status"] = [try JSONSerialization.data(withJSONObject: statusJSON)]
        let profitJSON: [String: Any] = [
            "balance_usd": "5000.0",
            "pnl_abs": "250.0",
            "pnl_pct": "5.0",
            "win_rate": 0.55
        ]
        client.responses["/profit"] = [try JSONSerialization.data(withJSONObject: profitJSON)]
        let equityJSON = [
            ["ts": Double(1_696_000_000_000), "equity": "5000.0"],
            ["ts": Double(1_696_006_000_000), "equity": "5250.0"]
        ]
        client.responses["/equity-curve"] = [try JSONSerialization.data(withJSONObject: equityJSON)]

        let repository = DashboardRepositoryImpl(apiClient: client, cache: DiskCache.shared)
        let viewModel = HomeViewModel(repository: repository)
        await viewModel.loadInitial()
        XCTAssertEqual(viewModel.status?.running, true)
        XCTAssertEqual(viewModel.profit?.balance, Decimal(string: "5000.0"))
        XCTAssertEqual(viewModel.equity.count, 2)
    }

    func testTradesPaginationMergesPages() async throws {
        let client = MockAPIClient()
        let firstPage: [String: Any] = [
            "items": [
                tradeJSON(id: "t1", pnl: 10, closedAt: 1_696_000_000_000),
                tradeJSON(id: "t2", pnl: -5, closedAt: 1_696_006_000_000)
            ],
            "next_cursor": "cursor"
        ]
        let secondPage: [String: Any] = [
            "items": [tradeJSON(id: "t3", pnl: 20, closedAt: 1_696_012_000_000)],
            "next_cursor": NSNull()
        ]
        client.responses["/trades"] = [
            try JSONSerialization.data(withJSONObject: firstPage),
            try JSONSerialization.data(withJSONObject: secondPage)
        ]

        let repository = TradesRepositoryImpl(apiClient: client, cache: DiskCache.shared)
        let viewModel = TradesViewModel(repository: repository)
        await viewModel.loadFirstPage()
        XCTAssertEqual(viewModel.items.count, 2)
        await viewModel.loadNextPageIfNeeded(currentIndex: 1)
        XCTAssertEqual(viewModel.items.count, 3)
    }

    func testCalendarViewModelComputesMonthTotals() async throws {
        let client = MockAPIClient()
        let tradesJSON: [String: Any] = [
            "items": [
                tradeJSON(id: "t1", pnl: 15, closedAt: 1_696_000_000_000),
                tradeJSON(id: "t2", pnl: -5, closedAt: 1_696_006_000_000)
            ],
            "next_cursor": NSNull()
        ]
        client.responses["/trades"] = [try JSONSerialization.data(withJSONObject: tradesJSON)]

        let repository = TradesRepositoryImpl(apiClient: client, cache: DiskCache.shared)
        let viewModel = CalendarViewModel(repository: repository)
        await viewModel.loadMonth(Date(timeIntervalSince1970: 1_696_000_000))
        XCTAssertEqual(viewModel.days.count, 2)
        XCTAssertEqual(viewModel.monthPnLTotal, 10)
    }

    private func tradeJSON(id: String, pnl: Double, closedAt: Double) -> [String: Any] {
        [
            "id": id,
            "symbol": "BTC/USD",
            "side": "buy",
            "qty": "1.0",
            "price": "1000.0",
            "pnl": "\(pnl)",
            "pnl_pct": "1.0",
            "opened_at": closedAt - 3_600_000,
            "closed_at": closedAt
        ]
    }
}
