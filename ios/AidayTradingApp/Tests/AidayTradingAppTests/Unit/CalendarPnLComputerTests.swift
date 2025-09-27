import XCTest
@testable import AidayTradingApp

final class CalendarPnLComputerTests: XCTestCase {
    private let timeZone = TimeZone(identifier: "America/Chicago")!

    func testAggregatesTradesByCentralTimeMidnight() {
        let trades = [
            Trade(
                id: "t1",
                symbol: "BTC/USD",
                side: "buy",
                quantity: 1,
                price: 1000,
                pnl: 100,
                pnlPercent: 1,
                openedAt: Date(timeIntervalSince1970: 1_696_000_000),
                closedAt: Date(timeIntervalSince1970: 1_696_003_000),
                timeZone: timeZone
            ),
            Trade(
                id: "t2",
                symbol: "ETH/USD",
                side: "sell",
                quantity: 1,
                price: 2000,
                pnl: -50,
                pnlPercent: -1,
                openedAt: Date(timeIntervalSince1970: 1_696_010_000),
                closedAt: Date(timeIntervalSince1970: 1_696_360_000),
                timeZone: timeZone
            )
        ]

        let results = CalendarPnLComputer.computeDailyPnL(trades: trades, in: timeZone)
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].pnlAbs, 100)
        XCTAssertEqual(results[1].pnlAbs, -50)
    }

    func testIgnoresOpenTrades() {
        let trades = [
            Trade(
                id: "t1",
                symbol: "BTC/USD",
                side: "buy",
                quantity: 1,
                price: 1000,
                pnl: 100,
                pnlPercent: 1,
                openedAt: Date(timeIntervalSince1970: 1_696_000_000),
                closedAt: nil,
                timeZone: timeZone
            )
        ]

        let results = CalendarPnLComputer.computeDailyPnL(trades: trades, in: timeZone)
        XCTAssertTrue(results.isEmpty)
    }

    func testNormalizesAcrossTimezoneBoundaries() {
        // Trade closes at 5 minutes past midnight Central Time (UTC-5/6 depending on DST)
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let components = DateComponents(year: 2023, month: 9, day: 1, hour: 5, minute: 5) // Equivalent to midnight CT during DST
        let utcDate = calendar.date(from: components)!

        let trade = Trade(
            id: "t1",
            symbol: "AAPL",
            side: "buy",
            quantity: 1,
            price: 190,
            pnl: 25,
            pnlPercent: 2,
            openedAt: utcDate,
            closedAt: utcDate,
            timeZone: timeZone
        )

        let results = CalendarPnLComputer.computeDailyPnL(trades: [trade], in: timeZone)
        XCTAssertEqual(results.count, 1)
        let day = results[0].date
        calendar.timeZone = timeZone
        XCTAssertEqual(calendar.component(.hour, from: day), 0)
        XCTAssertEqual(results[0].pnlAbs, 25)
    }
}
