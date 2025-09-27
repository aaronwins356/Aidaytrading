import XCTest
@testable import AidayTradingApp

@MainActor
final class CalendarDashboardViewModelTests: XCTestCase {
    func testGroupingResetsAtCentralMidnight() async throws {
        let service = MockReportingService()
        let formatter = ISO8601DateFormatter.apiFormatter
        let trades = [
            TradeRecord(
                id: 1,
                symbol: "ETHUSD",
                side: .buy,
                size: Decimal(string: "1.5")!,
                pnl: Decimal(string: "150.0")!,
                timestamp: formatter.date(from: "2024-05-15T04:30:00.000Z")!
            ),
            TradeRecord(
                id: 2,
                symbol: "ETHUSD",
                side: .sell,
                size: Decimal(string: "1.0")!,
                pnl: Decimal(string: "-50.0")!,
                timestamp: formatter.date(from: "2024-05-16T02:15:00.000Z")!
            )
        ]
        service.tradesResult = .success(TradesPage(items: trades, page: 1, pageSize: 200, total: trades.count))

        let viewModel = CalendarDashboardViewModel(month: formatter.date(from: "2024-05-01T00:00:00.000Z")!, accessToken: "token", reportingService: service)
        await viewModel.loadMonth()

        let positiveCell = viewModel.dayCells.first { cell in
            guard let date = cell.date else { return false }
            return Calendar(identifier: .gregorian).dateComponents([.day], from: date).day == 15
        }
        let negativeCell = viewModel.dayCells.first { cell in
            guard let date = cell.date else { return false }
            return Calendar(identifier: .gregorian).dateComponents([.day], from: date).day == 16
        }

        XCTAssertEqual(positiveCell?.pnl, Decimal(string: "150.0")!)
        XCTAssertEqual(negativeCell?.pnl, Decimal(string: "-50.0")!)
    }
}
