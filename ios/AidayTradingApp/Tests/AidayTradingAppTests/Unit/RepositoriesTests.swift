import XCTest
@testable import AidayTradingApp

final class RepositoriesTests: XCTestCase {
    private var session: URLSession!

    override func setUp() {
        super.setUp()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        session = URLSession(configuration: configuration)
        DiskCache.shared.removeValue(for: "profit_latest.json")
        DiskCache.shared.removeValue(for: "equity_latest.json")
        DiskCache.shared.removeValue(for: "status_latest.json")
    }

    func testFetchProfitCachesValue() async throws {
        MockURLProtocol.requestHandler = { request in
            guard request.url?.path == "/profit" else { throw URLError(.badURL) }
            let json: [String: Any] = [
                "balance_usd": "1200.0",
                "pnl_abs": "100.0",
                "pnl_pct": "5.0",
                "win_rate": 0.6
            ]
            let data = try JSONSerialization.data(withJSONObject: json)
            let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            return (response, data)
        }
        let client = APIClient(session: session, baseURL: URL(string: "https://example.com")!, interceptor: PassthroughInterceptor())
        let repository = DashboardRepositoryImpl(apiClient: client, cache: DiskCache.shared)
        let snapshot = try await repository.fetchProfit()
        XCTAssertEqual(snapshot.balance, Decimal(string: "1200.0"))
    }

    func testFetchProfitReturnsCacheOnFailure() async throws {
        try await testFetchProfitCachesValue()
        MockURLProtocol.requestHandler = { _ in throw URLError(.timedOut) }
        let client = APIClient(session: session, baseURL: URL(string: "https://example.com")!, interceptor: PassthroughInterceptor())
        let repository = DashboardRepositoryImpl(apiClient: client, cache: DiskCache.shared)
        let snapshot = try await repository.fetchProfit()
        XCTAssertEqual(snapshot.balance, Decimal(string: "1200.0"))
    }
}

final class PassthroughInterceptor: AuthIntercepting {
    func authorize(_ request: URLRequest) async throws -> URLRequest { request }
    func refreshAfterUnauthorized() async throws -> Bool { false }
}

final class MockURLProtocol: URLProtocol {
    static var requestHandler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let handler = MockURLProtocol.requestHandler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        do {
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
