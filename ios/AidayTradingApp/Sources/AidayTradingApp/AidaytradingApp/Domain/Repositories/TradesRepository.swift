import Foundation

protocol TradesRepository {
    func fetchTradesPage(limit: Int, cursor: String?) async throws -> (items: [Trade], nextCursor: String?)
    func fetchRecentTrades(limit: Int) async throws -> [Trade]
}
