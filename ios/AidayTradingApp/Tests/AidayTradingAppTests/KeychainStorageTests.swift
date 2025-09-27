import XCTest
@testable import AidayTradingApp

final class KeychainStorageTests: XCTestCase {
    func testRoundTripPersistsTokens() throws {
        #if canImport(Security)
        let storage = KeychainStorage()
        let tokens = AuthTokens(accessToken: "access-token", refreshToken: "refresh-token", accessTokenExpiry: Date().addingTimeInterval(3600))

        try storage.delete()
        try storage.save(tokens: tokens)
        let loaded = try storage.load()
        XCTAssertEqual(loaded?.accessToken, tokens.accessToken)
        XCTAssertEqual(loaded?.refreshToken, tokens.refreshToken)

        try storage.delete()
        XCTAssertNil(try storage.load())
        #else
        throw XCTSkip("Security framework unavailable on this platform")
        #endif
    }
}
