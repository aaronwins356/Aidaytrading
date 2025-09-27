import XCTest
@testable import AidayTradingApp

final class ReportingModelsTests: XCTestCase {
    func testProfitSummaryDecoding() throws {
        let json = """
        {
            "current_balance": "10500.50",
            "total_pl_amount": "500.50",
            "total_pl_percent": "5.005",
            "win_rate": 0.62
        }
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        let summary = try decoder.decode(ProfitSummary.self, from: json)
        XCTAssertEqual(summary.currentBalance, Decimal(string: "10500.50")!)
        XCTAssertEqual(summary.totalPLAmount, Decimal(string: "500.50")!)
        XCTAssertEqual(summary.totalPLPercent, Decimal(string: "5.005")!)
        XCTAssertEqual(summary.winRate, 0.62, accuracy: 0.0001)
    }

    func testEquityCurveDecoding() throws {
        let json = """
        [["2024-05-01T00:00:00.000Z", "10000.0"], ["2024-05-01T01:00:00.000Z", "10125.75"]]
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let points = try decoder.decode([EquityCurvePoint].self, from: json)
        XCTAssertEqual(points.count, 2)
        XCTAssertEqual(points[0].equity, Decimal(string: "10000.0")!)
        XCTAssertEqual(points[1].timestamp, ISO8601DateFormatter.apiFormatter.date(from: "2024-05-01T01:00:00.000Z"))
    }

    func testTradeRecordDecoding() throws {
        let json = """
        {
            "items": [
                {
                    "id": 1,
                    "symbol": "AAPL",
                    "side": "buy",
                    "size": "10",
                    "pnl": "15.5",
                    "timestamp": "2024-05-01T14:30:00.000Z"
                }
            ],
            "page": 1,
            "page_size": 50,
            "total": 1
        }
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let page = try decoder.decode(TradesPage.self, from: json)
        XCTAssertEqual(page.items.first?.symbol, "AAPL")
        XCTAssertEqual(page.items.first?.side, .buy)
        XCTAssertEqual(page.total, 1)
    }
}
