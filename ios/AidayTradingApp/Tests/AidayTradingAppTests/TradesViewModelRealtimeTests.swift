import Combine
import XCTest
@testable import AidayTradingApp

@MainActor
final class TradesViewModelRealtimeTests: XCTestCase {
    func testTradeExecutionAppendsToList() async throws {
        let initialTrade = Trade(
            id: "existing",
            symbol: "ETH/USD",
            side: "buy",
            quantity: 0.5,
            price: 1800,
            pnl: 50,
            pnlPercent: 0,
            openedAt: Date(),
            closedAt: Date(),
            timeZone: AppConfig.centralTimeZone
        )
        let repository = MockTradesRepository(pages: [(items: [initialTrade], nextCursor: nil)])
        let viewModel = TradesViewModel(repository: repository)
        let websocket = MockWebSocketClient()

        await viewModel.loadFirstPage()
        viewModel.attachRealtime(websocket)

        let newTrade = Trade(
            id: "t123",
            symbol: "BTC/USD",
            side: "sell",
            quantity: 0.2,
            price: 25_000,
            pnl: 320,
            pnlPercent: 0,
            openedAt: Date(),
            closedAt: Date(),
            timeZone: AppConfig.centralTimeZone
        )
        websocket.tradePublisher.send(newTrade)
        try await Task.sleep(nanoseconds: 20_000_000)

        XCTAssertEqual(viewModel.items.first?.id, "t123")
        XCTAssertTrue(viewModel.items.contains(where: { $0.id == "existing" }))
    }
}

private final class MockTradesRepository: TradesRepository {
    private let pages: [(items: [Trade], nextCursor: String?)]
    private var requestCount = 0

    init(pages: [(items: [Trade], nextCursor: String?)]) {
        self.pages = pages
    }

    func fetchTradesPage(limit: Int, cursor: String?) async throws -> (items: [Trade], nextCursor: String?) {
        let index = min(requestCount, pages.count - 1)
        let page = pages[index]
        requestCount += 1
        return page
    }

    func fetchRecentTrades(limit: Int) async throws -> [Trade] {
        pages.first?.items ?? []
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
