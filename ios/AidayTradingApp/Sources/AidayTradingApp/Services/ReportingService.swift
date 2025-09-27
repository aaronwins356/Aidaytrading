import Foundation

protocol ReportingServiceProtocol {
    func fetchSystemStatus(accessToken: String) async throws -> SystemStatus
    func fetchProfitSummary(accessToken: String) async throws -> ProfitSummary
    func fetchCurrentBalance(accessToken: String) async throws -> Decimal
    func fetchEquityCurve(accessToken: String, start: Date?, end: Date?, limit: Int?) async throws -> [EquityCurvePoint]
    func fetchTrades(
        accessToken: String,
        page: Int,
        pageSize: Int,
        symbol: String?,
        side: TradeSide?,
        start: Date?,
        end: Date?
    ) async throws -> TradesPage
}

final class ReportingService: ReportingServiceProtocol {
    private let apiClient: APIClientProtocol

    init(apiClient: APIClientProtocol = APIClient()) {
        self.apiClient = apiClient
    }

    func fetchSystemStatus(accessToken: String) async throws -> SystemStatus {
        try await apiClient.send(ReportingRequest.status(accessToken: accessToken), decode: SystemStatus.self)
    }

    func fetchProfitSummary(accessToken: String) async throws -> ProfitSummary {
        try await apiClient.send(ReportingRequest.profit(accessToken: accessToken), decode: ProfitSummary.self)
    }

    func fetchCurrentBalance(accessToken: String) async throws -> Decimal {
        let response = try await apiClient.send(ReportingRequest.balance(accessToken: accessToken), decode: BalanceSnapshot.self)
        return response.balance
    }

    func fetchEquityCurve(accessToken: String, start: Date?, end: Date?, limit: Int?) async throws -> [EquityCurvePoint] {
        try await apiClient.send(
            ReportingRequest.equityCurve(accessToken: accessToken, start: start, end: end, limit: limit),
            decode: [EquityCurvePoint].self
        )
    }

    func fetchTrades(
        accessToken: String,
        page: Int,
        pageSize: Int,
        symbol: String?,
        side: TradeSide?,
        start: Date?,
        end: Date?
    ) async throws -> TradesPage {
        try await apiClient.send(
            ReportingRequest.trades(
                accessToken: accessToken,
                page: page,
                pageSize: pageSize,
                symbol: symbol,
                side: side,
                start: start,
                end: end
            ),
            decode: TradesPage.self
        )
    }
}
