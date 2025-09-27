import SwiftUI
import XCTest
@testable import AidayTradingApp

#if canImport(SwiftUI)

@MainActor
final class CalendarViewSnapshotTests: XCTestCase {
    func testCalendarViewRendersWithPnLHeatmap() async throws {
        let repository = StubTradesRepository()
        let viewModel = CalendarViewModel(repository: repository)
        await viewModel.loadMonth(Date(timeIntervalSince1970: 1_696_000_000))
        let view = CalendarView(viewModel: viewModel)
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

private final class StubTradesRepository: TradesRepository {
    func fetchTradesPage(limit: Int, cursor: String?) async throws -> (items: [Trade], nextCursor: String?) {
        (items: sampleTrades, nextCursor: nil)
    }

    func fetchRecentTrades(limit: Int) async throws -> [Trade] {
        sampleTrades
    }

    private var sampleTrades: [Trade] {
        let tz = AppConfig.centralTimeZone
        return [
            Trade(id: "t1", symbol: "BTC/USD", side: "buy", quantity: 1, price: 1000, pnl: 50, pnlPercent: 2, openedAt: Date(timeIntervalSince1970: 1_696_000_000), closedAt: Date(timeIntervalSince1970: 1_696_000_000), timeZone: tz),
            Trade(id: "t2", symbol: "ETH/USD", side: "sell", quantity: 1, price: 1500, pnl: -20, pnlPercent: -1, openedAt: Date(timeIntervalSince1970: 1_696_050_000), closedAt: Date(timeIntervalSince1970: 1_696_050_000), timeZone: tz)
        ]
    }
}

#endif
