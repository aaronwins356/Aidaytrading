import Foundation

final class TradesRepositoryImpl: TradesRepository {
    private struct TradesResponse: Codable {
        let items: [TradeDTO]
        let nextCursor: String?

        private enum CodingKeys: String, CodingKey {
            case items
            case nextCursor = "next_cursor"
        }
    }

    private let apiClient: APIClientPerforming
    private let cache: DiskCache
    private let timeZone: TimeZone

    init(
        apiClient: APIClientPerforming = APIClient(),
        cache: DiskCache = .shared,
        timeZone: TimeZone = AppConfig.centralTimeZone
    ) {
        self.apiClient = apiClient
        self.cache = cache
        self.timeZone = timeZone
    }

    func fetchTradesPage(limit: Int, cursor: String?) async throws -> (items: [Trade], nextCursor: String?) {
        do {
            let response: TradesResponse = try await apiClient.get(Endpoints.trades(limit: limit, cursor: cursor))
            let trades = response.items.map { $0.toEntity(timeZone: timeZone) }
            if cursor == nil {
                try cache.save(response, for: "trades_page_1.json")
            }
            return (trades, response.nextCursor)
        } catch {
            if cursor == nil, let cached: TradesResponse = cache.load(TradesResponse.self, for: "trades_page_1.json") {
                return (cached.items.map { $0.toEntity(timeZone: timeZone) }, cached.nextCursor)
            }
            throw error
        }
    }

    func fetchRecentTrades(limit: Int) async throws -> [Trade] {
        let response = try await fetchTradesPage(limit: limit, cursor: nil)
        return response.items
    }
}
