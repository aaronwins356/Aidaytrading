import XCTest
@testable import AidayTradingApp

final class CurrencyFormattingTests: XCTestCase {
    func testCurrencyStringFormatsUSD() {
        let value = Decimal(string: "12345.67")!
        let formatted = value.currencyString()
        XCTAssertTrue(formatted.contains("12"))
        XCTAssertTrue(formatted.contains("$") || formatted.contains("US$"))
    }

    func testSignedPercentStringAddsSign() {
        let positive = Decimal(12.345)
        let negative = Decimal(-2.5)
        XCTAssertTrue(positive.signedPercentString().contains("+"))
        XCTAssertTrue(negative.signedPercentString().contains("−"))
    }
}
